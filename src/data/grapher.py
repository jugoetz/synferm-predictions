from typing import Optional, Union, Dict, Tuple, List

import dgl
import numpy as np
import torch
from dgllife.utils.mol_to_graph import mol_to_bigraph


from rdkit.Chem import MolFromSmiles
from rdkit.Chem.rdchem import Mol


def build_mol_graph(
    mol: Union[str, Mol],
    atom_featurizer: callable,
    bond_featurizer: callable,
    global_featurizer: Optional[callable] = None,
    graph_type: str = "bond_edges",
) -> dgl.DGLGraph:
    """
    Build a graph from a SMILES representing a molecule.

    The resulting graph is a chemical graph where atoms are encoded as nodes and bonds are encoded as edges.

    Args:
        mol (str, Mol): SMILES or Mol representing the molecule
        atom_featurizer (callable): Takes a Mol and returns a dict of features
        bond_featurizer (callable): Takes a Mol and returns a dict of features
        global_featurizer (optional): Not implemented. Raises an exception if anything is passed.
        graph_type (str): Type of graph to use. If "bond_edges", graphs are formed as molecular graphs (nodes are
                    atoms and edges are bonds). These graphs are homogeneous. If "bond_nodes", bond-node graphs will be
                    formed (both atoms and bonds are nodes, edges represent their connectivity).
                    Options: {"bond_edges", "bond_nodes"}. Default "bond_edges".

    Returns:
        dgl.DGLGraph: Graph-encoding of smiles, with features as provided by featurizers. Nodes represent atoms and
            edges represent bonds.
    """
    if isinstance(mol, str):
        mol = MolFromSmiles(mol)

    g = mol_to_bigraph(
        mol,
        node_featurizer=atom_featurizer,
        edge_featurizer=bond_featurizer,
        canonical_atom_order=False,
    )

    if global_featurizer:
        # could calculate some global feature here and attach it to the graph
        raise NotImplementedError

    if graph_type == "bond_edges":
        return g
    elif graph_type == "bond_nodes":
        # as the last step, we transform "bond edges" representation of the CGR to our "bond nodes" representation
        bn_graph = bond_edges_to_bond_nodes(g)
        return bn_graph
    else:
        raise ValueError("Invalid value for argument 'graph_type'")


def build_cgr(
    reaction_smiles: str,
    atom_featurizer: callable,
    bond_featurizer: callable,
    global_featurizer: Optional[callable] = None,
    mode: str = "reac_diff",
    graph_type: str = "bond_edges",
) -> dgl.DGLHeteroGraph:
    """
    Build CGR (condensed graph of reaction) for a reaction and featurize nodes.

    The flavor of CGR returned by this function differs from the one used in Heid and Green
    (https://doi.org/10.1021/acs.jcim.1c00975).
    The CGR returned here is a heterograph with two types of nodes:
        - "atom"
        - "bond"
    and four types of (directed) edges:
        - ("bond", "starts_at", "atom")
        - ("atom", "starts", "bond")
        - ("bond", "leads_to", "bond")
        - ("bond", "leads_to", "atom")

    Args:
        reaction_smiles (str): atom-mapped reactionSMILES string, may be unbalanced.
        atom_featurizer (callable): Mol -> dict, returns a dict containing atom features
        bond_featurizer (callable): Mol -> dict, returns a dict containing bod features
        global_featurizer (optional): Featurizer to generate global node features.[NOT SUPPORTED, i.e. must be None]
        mode (str): How to construct CGR features:
                - "reac_prod": Concatenate reactant and product features.
                - "reac_diff" (default): Concatenate reactant features and (f_prod - f_reac)
                - "prod_diff": Concatenate product features and (f_reac - f_prod)
        graph_type (str): Type of graph to use. If "bond_edges", graphs are formed as molecular graphs (nodes are
                    atoms and edges are bonds). These graphs are homogeneous. If "bond_nodes", bond-node graphs will be
                    formed (both atoms and bonds are nodes, edges represent their connectivity).
                    Options: {"bond_edges", "bond_nodes"}. Default "bond_edges".

    Returns:
        dgl.DGLHeteroGraph: CGR for the reaction.
    """
    if global_featurizer is not None:
        raise NotImplementedError("global_featurizer is not implemented. Pass 'None'.")

    try:
        # extract reactant/product Mol (i.e. a single Mol for all reactants and one for all products)
        reactants = MolFromSmiles(reaction_smiles.split(">>")[0])
        products = MolFromSmiles(reaction_smiles.split(">>")[1])

        # create the CGR (as defined in Heid et al.)
        cgr_g = create_cgr(reactants, products, atom_featurizer, bond_featurizer, mode)

    except Exception as e:
        raise ValueError(f"Could not build CGR for reaction: {reaction_smiles}") from e

    if graph_type == "bond_edges":
        return cgr_g
    elif graph_type == "bond_nodes":
        # as the last step, we transform "bond edges" representation of the CGR to our "bond nodes" representation
        bn_graph = bond_edges_to_bond_nodes(cgr_g)
        return bn_graph
    else:
        raise ValueError("Invalid value for argument 'graph_type'")


def bond_edges_to_bond_nodes(graph: dgl.DGLGraph) -> dgl.DGLHeteroGraph:
    """
    Get a BN graph from a BE graph

    A graph implementation suitable for message passing would have bond nodes (two nodes for each bond, because
    we want to pass directed messages). It could still have atom nodes, which however would only be connected
    to bond nodes and not other atom nodes.

    The resulting heterograph would have (bond, leads_to, bond) and
    (bond, starts_at, atom) edges. All (chemical) structural information would be captured in the
    (bond, leads_to, bond) edges.

    One might think that a (bond, is_reverse_of, bond) edge would be necessary as well, but that
    is not the case. The structure of the (bond, leads_to, bond) subgraph already precludes the
    adverse case that information is propagated in cyclic manner among a bond and its reverse by the simple
    definition that a bond cannot lead to its own reverse,
    i.e., a (bond_vw, leads_to, bond_wv) edge cannot exist.

    The (bond, starts_at, atom) edges contain all information necessary to construct the atom-centered messages
    (step 3 of the D-MPNN). It is convenient to add the inverse of these edges, so (atom, starts, bond), because
    it allows simple message flow in the instantiation phase (it would not be strictly necessary though).

    It is further convenient to add (bond, leads_to, atom) edges, to be able to recover the original structure of
    the chemical graph (e.g. for transforms). The original bidirectional atom-only graph structure can be recovered from
    the BN graph by adding bond edges given by the meta path {bond: [(atom, starts, bond), (bond, leads_to, atom)]}.

    Args:
        graph (dgl.DGLGraph): A graph with atoms as nodes and bonds as edges

    Returns:
        dgl.DGLHeteroGraph: A hetero-graph with two node types "atom" and bond"
    """
    # first we add a node for every bond edge
    start, end, bond = graph.edges(form="all")
    num_bonds = len(bond)

    # we generate a tuple of tensors giving the (bond, leads_to, bond) edges
    # this still contains connections to the reverse edge
    # it works by expanding both start and end node tensors to square shape, then transposing end
    # this way we span a "correlation grid" of bonds and all head to tail connections give True
    # The reverse bonds will be the ones that are symmetric with respect to the diagonal
    # by simply AND-NOTing the transpose we can get rid of these symmetric elements
    bond_connection_matrix = (
        start.expand(num_bonds, num_bonds) == end.expand(num_bonds, num_bonds).T
    )
    bond_leads_to_bond = torch.where(
        bond_connection_matrix & (~bond_connection_matrix.T)
    )

    bn_graph_data = {
        ("bond", "starts_at", "atom"): (bond, start),
        ("atom", "starts", "bond"): (
            start,
            bond,
        ),  # reverse of the above, useful to control message-flow
        ("bond", "leads_to", "bond"): bond_leads_to_bond,
        ("bond", "leads_to", "atom"): (bond, end),  # redundant, but convenient
    }

    bn_graph = dgl.heterograph(bn_graph_data)
    # copy properties from original graph
    bn_graph.nodes["atom"].data["x"] = graph.ndata["x"]
    bn_graph.nodes["bond"].data["e"] = graph.edata["e"]

    return bn_graph


def get_atom_map_numbers(mol: Mol) -> list:
    """
    Get the "molAtomMapNumber" for all atoms in a molecule.

    Args:
        mol (Mol): rdkit molecule

    Returns:
        list: List of atom map numbers, ordered by atom indices. If the property "molAtomMapNumber" was not set for
            atom i, then the list will contain None at index i.
    """

    map_nos = []
    for a in mol.GetAtoms():
        if a.HasProp("molAtomMapNumber"):
            map_nos.append(a.GetProp("molAtomMapNumber"))
        else:
            map_nos.append(None)
    return map_nos


def get_atom_to_bond_maps(mol: Mol) -> Dict[Tuple[int, int], int]:
    """
    Get the mapping from atom indices (NOT atom map numbers) to bond indices

    Args:
        mol (Mol): rdkit molecule

    Returns:
        dict: Containing elements of the form `(begin_atom_idx, end_atom_idx): bond_idx`
    """
    a2b = {}
    for i, b in enumerate(mol.GetBonds()):
        a2b[(b.GetBeginAtomIdx(), b.GetEndAtomIdx())] = i
    return a2b


def create_cgr(
    reactants: Mol,
    products: Mol,
    atom_featurizer: callable,
    bond_featurizer: callable,
    mode: str = "reac_diff",
) -> dgl.DGLGraph:
    """
    Create the condensed graph of reaction (CGR) from a graph containing reactants and a graph containing products.

    The reaction does not have to be balanced, but atom-mapped.

    Args:
        reactants (Mol): rdkit molecule of all reactants
        products (Mol): rdkit molecule of all products
        atom_featurizer (callable):
        bond_featurizer (callable):
        mode (str): How to construct CGR features:
                - "reac_prod": Concatenate reactant and product features.
                - "reac_diff" (default): Concatenate reactant features and (f_prod - f_reac)
                - "prod_diff": Concatenate product features and (f_reac - f_prod)

    Returns:
        dgl.DGLGraph: CGR of the reaction as homogeneous graph
    """

    # extract atom map numbers
    reactant_atom_map_numbers = get_atom_map_numbers(reactants)
    product_atom_map_numbers = get_atom_map_numbers(products)

    # extract atom-to-bond relations
    reactant_a2b_mapping = get_atom_to_bond_maps(reactants)
    product_a2b_mapping = get_atom_to_bond_maps(products)

    # get the relations of the reactant and product indices
    (
        r2p_atoms,
        ronly_atoms,
        ponly_atoms,
        r2p_bonds,
        ronly_bonds,
        ponly_bonds,
    ) = atom_and_bond_mapping(
        reactant_atom_map_numbers,
        product_atom_map_numbers,
        reactant_a2b_mapping,
        product_a2b_mapping,
    )

    reactant_b2a = {v: k for k, v in reactant_a2b_mapping.items()}
    product_b2a = {v: k for k, v in product_a2b_mapping.items()}
    atoms_p2r = {j: i for i, j in r2p_atoms}
    # these will hold data for the graph
    node_src_data = []
    node_dst_data = []
    # this will record the relation between CGR bonds and reactants and product (to let us assign features easily)
    cgr_b2rp = []
    # to construct the CGR, we first need the union of all bonds, unchanged and changed
    # first, we add all edges that are in both reactants and products. We use the reactant atom indices.
    for r_b_idx, p_b_idx in r2p_bonds:
        src_atom, dst_atom = reactant_b2a[r_b_idx]
        node_src_data.append(src_atom)
        node_dst_data.append(dst_atom)
        cgr_b2rp.append((r_b_idx, p_b_idx))
    # then we add edges only in reactants (i.e. lost bonds), following the same logic
    for r_b_idx in ronly_bonds:
        src_atom, dst_atom = reactant_b2a[r_b_idx]
        node_src_data.append(src_atom)
        node_dst_data.append(dst_atom)
        cgr_b2rp.append((r_b_idx, None))
    # the cgr_a2rp relation so far follows from the logic we used in the CGR construction
    cgr_a2rp = sorted(
        [(r_a, None) for r_a in ronly_atoms] + r2p_atoms, key=lambda x: x[0]
    )
    # finally, we add edges that are only in products (i.e. formed bonds).
    # Here, we need to translate from product atom indices to reactant atom indices to stay consistent.
    # If the atom is not in reactants, we create a new (higher) index.
    for p_b_idx in ponly_bonds:
        src_atom_p, dst_atom_p = product_b2a[p_b_idx]
        try:  # exception if atom is not in reactants
            src_atom = atoms_p2r[src_atom_p]
        except KeyError:
            # we assign a new atom index that is 1 higher than the index of any atom that is in reactants
            src_atom = 1 + max(node_src_data + node_dst_data)
            cgr_a2rp.append(
                (src_atom, src_atom_p)
            )  # we need to add the newly created reactant atom index to our relations
        node_src_data.append(src_atom)
        try:  # exception if atom is not in reactants
            dst_atom = atoms_p2r[dst_atom_p]
        except KeyError:
            # we assign a new atom index that is 1 higher than the index of any atom that is in reactants
            # we already appended src_atom to node_src_data, so we have that covered as well
            dst_atom = 1 + max(node_src_data + node_dst_data)
            cgr_a2rp.append(
                (dst_atom, dst_atom_p)
            )  # we need to add the newly created reactant atom index to our relations
        node_dst_data.append(dst_atom)

        cgr_b2rp.append((None, p_b_idx))

    # We require the CGR to have the reverse edge for every bond edge in an order such that bond i gives rise to
    # edges (2i) and (2i + 1) in the graph.
    node_src = torch.zeros(2 * len(node_src_data), dtype=torch.int32)
    node_dst = torch.zeros(2 * len(node_src_data), dtype=torch.int32)

    for i, v in enumerate(node_src_data):
        node_src[2 * i] = v
        node_dst[2 * i + 1] = v
    for i, v in enumerate(node_dst_data):
        node_src[2 * i + 1] = v
        node_dst[2 * i] = v

    # now we have the structure of the graph ready. We want the features in the same way
    # the featurizers are expected to return a dict {"feat_name": features}, where features is a torch.tensor
    # of shape(N, feature_length) with N=n_atoms or N=(2 * n_bonds), respectively. The order must be the index order
    # of the rdkit molecule, and for bonds every bond index i must be related to features[2i] and features[2i+1]

    r_a_features = atom_featurizer(reactants)
    r_b_features = bond_featurizer(reactants)
    p_a_features = atom_featurizer(products)
    p_b_features = bond_featurizer(products)

    # we merge the atom features using the reactant indices (which we used for CGR creation)
    cgr_a_feats = {}
    for k, r_feat in r_a_features.items():
        feature_len = r_feat.shape[1]
        n_atoms = len(cgr_a2rp)
        p_feat = p_a_features[k]
        # instantiate 0-padded tensors to hold ordered feats
        r_feat_padded = torch.zeros((n_atoms, feature_len))
        p_feat_padded = torch.zeros((n_atoms, feature_len))
        for i, (r_idx, p_idx) in enumerate(cgr_a2rp):
            if r_idx:
                r_feat_padded[i] = r_feat[r_idx]
            if p_idx:
                p_feat_padded[i] = p_feat[p_idx]

        cgr_feat = torch.zeros((n_atoms, feature_len * 2))
        if mode == "reac_diff":
            cgr_feat[:, :feature_len] = r_feat_padded
            cgr_feat[:, feature_len:] = p_feat_padded - r_feat_padded
        elif mode == "reac_prod":
            cgr_feat[:, :feature_len] = r_feat_padded
            cgr_feat[:, feature_len:] = p_feat_padded
        elif mode in ["prod_diff"]:
            cgr_feat[:, :feature_len] = p_feat_padded
            cgr_feat[:, feature_len:] = r_feat_padded - p_feat_padded
        else:
            raise ValueError(f"Parameter 'mode' cannot take value '{mode}'")

        cgr_a_feats[k] = cgr_feat

    # we merge the bond features using the same logic as for atoms
    cgr_b_feats = {}
    for k, r_feat in r_b_features.items():
        feature_len = r_feat.shape[1]
        n_bonds = len(cgr_b2rp)
        p_feat = p_b_features[k]
        # instantiate 0-padded tensors to hold ordered feats
        r_feat_padded = torch.zeros((2 * n_bonds, feature_len))
        p_feat_padded = torch.zeros((2 * n_bonds, feature_len))
        for i, (r_idx, p_idx) in enumerate(cgr_b2rp):
            if r_idx:
                r_feat_padded[2 * i] = r_feat[2 * r_idx]
                r_feat_padded[2 * i + 1] = r_feat[2 * r_idx + 1]
            if p_idx:
                p_feat_padded[2 * i] = p_feat[2 * p_idx]
                p_feat_padded[2 * i + 1] = p_feat[2 * p_idx + 1]

        cgr_feat = torch.zeros((n_bonds * 2, feature_len * 2))
        if mode == "reac_diff":
            cgr_feat[:, :feature_len] = r_feat_padded
            cgr_feat[:, feature_len:] = p_feat_padded - r_feat_padded
        elif mode == "reac_prod":
            cgr_feat[:, :feature_len] = r_feat_padded
            cgr_feat[:, feature_len:] = p_feat_padded
        elif mode in ["prod_diff"]:
            cgr_feat[:, :feature_len] = p_feat_padded
            cgr_feat[:, feature_len:] = r_feat_padded - p_feat_padded
        else:
            raise ValueError(f"Parameter 'mode' cannot take value '{mode}'")

        cgr_b_feats[k] = cgr_feat

    # put everything together in a graph
    cgr_g = dgl.graph((node_src, node_dst))
    for k, v in cgr_a_feats.items():
        cgr_g.ndata[k] = v
    for k, v in cgr_b_feats.items():
        cgr_g.edata[k] = v

    return cgr_g


def atom_and_bond_mapping(
    reactant_atom_map_numbers: List[int],
    product_atom_map_numbers: List[int],
    reactant_a2b_mapping: Dict[Tuple[int, int], int],
    product_a2b_mapping: Dict[Tuple[int, int], int],
) -> Tuple[
    List[Tuple[int, int]], int, int, List[Tuple[int, int]], List[int], List[int]
]:
    """
    Map atoms and bonds between reactants and products

    Args:
        reactant_atom_map_numbers (list): molAtomMapNumbers in order of the atom indices, for reactant mol
        product_atom_map_numbers (list): molAtomMapNumbers in order of the atom indices, for product mol
        reactant_a2b_mapping (dict): Keys are tuples of atom indices indicating start and end of the bond, value is the bond index. For rectants.
        product_a2b_mapping (dict): Keys are tuples of atom indices indicating start and end of the bond, value is the bond index. For products.

    Returns:
        Tuple[list]:
            Mapping from reactant atom indices to product atom indices,
            Atom indices only in reactants,
            Atom indices only in products,
            Mapping from reactant bond indices to product atom indices,
            Bond indices only in reactants,
            Bond indices only in products,

    """

    reactant_atom_map_arr = np.array(
        reactant_atom_map_numbers, dtype=float
    )  # None -> np.nan
    product_atom_map_arr = np.array(product_atom_map_numbers, dtype=float)

    # atoms
    reac_atom_idx = np.where(~np.isnan(reactant_atom_map_arr))[
        0
    ]  # indices of reactant atoms that end up in products
    reac2prod_atom_idx = [
        (i, j)
        for i, j in zip(
            reac_atom_idx,
            find_indices(product_atom_map_arr, reactant_atom_map_arr[reac_atom_idx]),
        )
    ]  # tuples (reactant_atom_idx, corresponding_product_atom_idx)
    reac2prod_atom_idx_map = {
        i: j for i, j in reac2prod_atom_idx
    }  # same as above, but as dict
    reaconly_atom_idx = np.where(np.isnan(reactant_atom_map_arr))[
        0
    ].tolist()  # atom indices of reactant atoms not in products
    prodonly_atom_idx = np.where(np.isnan(product_atom_map_arr))[
        0
    ].tolist()  # atom indices of product atoms not in reactants

    # bonds
    reac2prod_bond_idx = []
    reaconly_bond_idx = []
    prodonly_bond_idx = []
    for k, v in reactant_a2b_mapping.items():
        try:  # KeyError if the bond is not in both products and reactants
            reac2prod_bond_idx.append(
                (
                    v,
                    product_a2b_mapping[
                        (reac2prod_atom_idx_map[k[0]], reac2prod_atom_idx_map[k[1]])
                    ],
                )
            )
        except KeyError:
            reaconly_bond_idx.append(v)

    conserved_prod_bonds = [b[1] for b in reac2prod_bond_idx]
    for v in product_a2b_mapping.values():
        if v not in conserved_prod_bonds:
            prodonly_bond_idx.append(v)

    return (
        reac2prod_atom_idx,
        reaconly_atom_idx,
        prodonly_atom_idx,
        reac2prod_bond_idx,
        reaconly_bond_idx,
        prodonly_bond_idx,
    )


def find_indices(arr, values):
    """
    Find index of v in arr for every value v in values.

    Requires that all values that are both in values and in arr are unique in arr.
    """
    idx = []
    for v in values:
        idx.append(np.where(arr == v)[0])
    return np.concatenate(idx)

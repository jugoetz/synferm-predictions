import warnings
from typing import Union

from rdkit import Chem
from rdkit.Chem import Draw, rdChemReactions
from rdkit.Chem.SaltRemover import SaltRemover
from rdkit.Chem.rdchem import Mol


def desalt_building_block(mol: Union[str, Chem.Mol]) -> Chem.Mol:
    def deprotonate_nitrogen(mol):
        """Remove a proton from ammonium species"""
        mol.UpdatePropertyCache()
        patt = Chem.MolFromSmarts(
            "[#7+;H1,H2,H3,h1,h2,h3]"
        )  # this pattern matches positive N with at least one proton attached
        try:
            idx = mol.GetSubstructMatches(patt)[0][
                0
            ]  # this raises IndexError if patt is not found
            atom = mol.GetAtomWithIdx(idx)  # get the atom index of the charged N
            atom.SetFormalCharge(0)
            """
            If H are explicit, we have to do explicit removal. If they are implicit, calling UpdatePropertyCache() suffices
            """
            n_hyd = atom.GetNumExplicitHs()
            if n_hyd > 0:
                n_hyd -= 1
                atom.SetNumExplicitHs(n_hyd)
            mol.UpdatePropertyCache()
        except IndexError:
            pass

        return None

    if isinstance(mol, str):
        mol = Chem.MolFromSmiles(mol)
    else:
        mol = Chem.Mol(mol)
    # desalt the building block library
    remover = SaltRemover()
    mol_desalt = remover.StripMol(mol)
    # neutralize ammoniums
    deprotonate_nitrogen(mol_desalt)
    return mol_desalt


def remove_atom_map_numbers(mol: Mol) -> None:
    """
    Remove atom-mapping numbers from a molecule. Operates in place.
    """
    for atom in mol.GetAtoms():
        if atom.HasProp("molAtomMapNumber"):
            atom.ClearProp("molAtomMapNumber")
    return


def canonicalize_smiles(smiles: str, remove_explicit_H: bool = False) -> str:
    """
    Canonicalize a SMILES string.

    Removes any atom-mapping numbers. Optionally, removes explicit Hs.
    """
    mol = Chem.MolFromSmiles(smiles)
    for a in mol.GetAtoms():
        if a.HasProp("molAtomMapNumber"):
            a.ClearProp("molAtomMapNumber")
    if remove_explicit_H:
        mol = Chem.RemoveHs(mol)
    return Chem.MolToSmiles(mol)


def standardize_building_block(
    smiles: str, remove_map_no: bool = False, return_smiles=False
) -> Union[str, Mol]:
    """
    Standardize a building block. This can be used to sanitize inputs to the ML models.

    Includes desalting, taking the largest fragment, standardizing stereochemistry of the monomer protecting group.

    For initiators, this will result give an anion, stripped of the K+ counterion.
    For monomers, this will result in a neutral, desalted molecule.
    The stereochemistry of the spiro carbon of the protecting group will be set to undefined
    (b/c we do not expect an influence on reactivity and when we generate reactants from products for inference,
     we cannot know the configuration of that stereocenter).
    For terminators, this will result in a neutral, desalted molecule.

    Molecules standardized with this function can be used as inputs to all featurizers in this project.
    If a SMILES string is needed, e.g. for the OneHotEncoder, pass the returned molecule to Chem.MolToSmiles().

    Args:
        smiles (str): The SMILES string of the building block
        remove_map_no (bool): Whether to remove atom-mapping numbers.
            Set to true it input is taken from atom-mapped reactionSMILES.
            Default: False
        return_smiles (bool): Whether to return the standardized molecule as SMILES string. Default: False
    Returns:
        str or Mol: The standardized molecule
    """
    if remove_map_no:
        mol_or_smiles = Chem.MolFromSmiles(smiles)
        remove_atom_map_numbers(mol_or_smiles)
    else:
        mol_or_smiles = smiles
    mol = remove_monomer_pg_chirality(desalt_building_block(mol_or_smiles))

    if return_smiles:
        return Chem.MolToSmiles(mol)
    else:
        return mol


def move_atom_index_to_mapno(mol: Mol):
    """
    Write the atom indexes to property "molAtomMapNo", so they can be displayed in drawing.

    Note that this overwrites any previous "molAtomMapNo".
    """
    mol_copy = Chem.Mol(mol)  # necessary to not change the input molecule
    for i, a in enumerate(mol_copy.GetAtoms()):
        if a.HasProp("molAtomMapNumber"):
            a.ClearProp("molAtomMapNumber")
        a.SetProp("molAtomMapNumber", str(i))
    return mol_copy


def move_bond_index_to_props(mol: Mol):
    """
    Write the bond indexes to property "bondIndex", so they can be displayed in drawing.
    """
    mol_copy = Chem.Mol(mol)  # necessary to not change the input molecule
    for i, b in enumerate(mol_copy.GetBonds()):
        b.SetProp("bondNote", str(i))
    return mol_copy


def mol_to_file_with_indices(mol: Mol, file: str):
    """
    Draw a molecule to file with indices for atoms and bonds displayed
    """
    mol = move_atom_index_to_mapno(mol)
    mol = move_bond_index_to_props(mol)
    Draw.MolToFile(mol, file)


def remove_mapno(mol: Mol):
    """
    Remove all atom-mapping numbers from a molecule.
    """
    mol_copy = Chem.Mol(mol)
    for atom in mol_copy.GetAtoms():
        if atom.HasProp("molAtomMapNumber"):
            atom.ClearProp("molAtomMapNumber")
    return mol_copy


def remove_mapno_from_reaction(rxn: rdChemReactions.ChemicalReaction) -> None:
    """
    Remove all atom-mapping information from a reaction. Operates inplace.
    """
    for ri in range(rxn.GetNumReactantTemplates()):
        rt = rxn.GetReactantTemplate(ri)
        for atom in rt.GetAtoms():
            if atom.HasProp("molAtomMapNumber"):
                atom.ClearProp("molAtomMapNumber")
    for pi in range(rxn.GetNumProductTemplates()):
        pt = rxn.GetProductTemplate(pi)
        for atom in pt.GetAtoms():
            if atom.HasProp("molAtomMapNumber"):
                atom.ClearProp("molAtomMapNumber")
    return


def create_reaction_instance(rxn, reactants):
    """
    Create an instance of a reaction, given reactants, and map all atoms that end up in the product(s).
    This is adapted from Greg's code in https://github.com/rdkit/rdkit/issues/1269#issuecomment-275228746,
    but extended to map the side chains as well.
    Copied from https://github.com/jugoetz/slap-platform-predict/blob/5acb77ff4fdc412f7fd03a8226a4635e086978d7/src/util/rdkit_util.py#L82
    Note that atoms that are not present in the product (unbalanced reaction equation) will not be annotated.
    """

    # first, we set a tag on reactant atoms. This will be passed on to the product for all non-mapped atoms
    for i, sm in enumerate(reactants):
        for atom in sm.GetAtoms():
            atom.SetProp("tag", "reactant-%s atom-%s" % (i, atom.GetIdx()))

    # for the mapped atoms, extract their mapping in the reactants
    map_number_to_reactant = {}
    for i, reactant in enumerate(rxn.GetReactants()):
        for atom in reactant.GetAtoms():
            if atom.HasProp("molAtomMapNumber"):
                map_number_to_reactant[atom.GetIntProp("molAtomMapNumber")] = (
                    i,
                    atom.GetIdx(),
                )

    mapped_reactions = []  # this will hold the reactions
    product_set = rxn.RunReactants(reactants)  # run the reaction to get product set
    if len(product_set) == 2:
        # if one of the reagents is cyclic?or a ketone? we get two identical product sets. We remove one of them.
        if Chem.MolToSmiles(product_set[0][0]) == Chem.MolToSmiles(product_set[1][0]):
            product_set = [
                product_set[0],
            ]

    # now, we look into the products
    for products in product_set:
        # we need to know the highest mapno, because mapping the "tagged" atoms will have to start above that
        mapno_max = max(
            map_number_to_reactant.keys()
        )  # needs to reset on every product_set
        reactant_list = [Chem.Mol(x) for x in reactants]
        reaction = rdChemReactions.ChemicalReaction()
        for p in products:
            for atom in p.GetAtoms():
                # for atoms that are mapped in the reaction template
                # (RDKit does not copy user-defined properties to the products for mapped atoms, only for unmapped atoms)
                # the reference solution from Greg uses "if not atom.HasProp('old_mapno'):" here, but that is too wide
                # and fails for cyclic ketones, where the carbonyl C will be mapped twice.
                if not atom.HasProp("tag"):
                    mno = atom.GetIntProp("old_mapno")
                    atom.SetIntProp("molAtomMapNumber", mno)
                    ridx, aidx = map_number_to_reactant[mno]
                    # aidx is the index of the atom in the reactant template. We need
                    # to read out the number in the actual reactant:
                    raidx = atom.GetIntProp("react_atom_idx")
                    ratom = (
                        reactant_list[ridx]
                        .GetAtomWithIdx(raidx)
                        .SetIntProp("molAtomMapNumber", mno)
                    )

                # for atoms that are unmapped in the reaction template
                else:
                    tag = atom.GetProp("tag")
                    mapno_max += 1
                    atom.SetIntProp("molAtomMapNumber", mapno_max)
                    # now find the tag in reactant_list
                    for sm in reactant_list:
                        for ratom in sm.GetAtoms():
                            if ratom.HasProp("tag"):
                                if ratom.GetProp("tag") == tag:
                                    ratom.SetIntProp("molAtomMapNumber", mapno_max)

            # now add the product to the reaction
            reaction.AddProductTemplate(p)
        # add the reactants to reaction
        for reactant in reactant_list:
            reaction.AddReactantTemplate(reactant)
        # add reaction for all product sets
        mapped_reactions.append(reaction)
    return mapped_reactions


def map_reactions(rxn, reactant_sets, error_level="warn"):
    """
    Take a reaction template and a list of reactant sets and return the mapped reactions.
    Adapted from https://github.com/jugoetz/slap-platform-predict/blob/5acb77ff4fdc412f7fd03a8226a4635e086978d7/src/util/rdkit_util.py#L163

    Args:
        rxn: RDKit reaction object
        reactant_sets: list of reactant sets
        error_level: "error" or "warn" - whether to raise RuntimeError or just warn if mapping a reaction fails.

    Returns:
        list of mapped reactions
    """
    mapped_reactions = []
    for i, reactant_set in enumerate(reactant_sets):
        reaction_inst = create_reaction_instance(rxn, reactant_set)
        if len(reaction_inst) == 1:  # all good
            mapped_reactions.append(reaction_inst[0])
        elif len(reaction_inst) == 0:  # failed
            mapped_reactions.append(None)
            if error_level == "error":
                raise RuntimeError(f"ERROR: No product for reactant set with index {i}")
            elif error_level == "warn":
                warnings.warn(f"ERROR: No product for reactant set with index {i}")
            else:
                raise ValueError(f"Unknown error level: {error_level}")
        else:  # too many resulting reactions
            # remove any duplicates
            idx = []
            unique_reactions = []
            for j, reac in enumerate(reaction_inst):
                reac_smarts = rdChemReactions.ReactionToSmarts(reac)
                if reac_smarts not in unique_reactions:
                    idx.append(j)
                    unique_reactions.append(reac_smarts)
                reaction_inst_cleaned = [reaction_inst[j] for j in idx]
                if len(reaction_inst_cleaned) > 1:
                    if error_level == "error":
                        raise RuntimeError(
                            f"ERROR: Multiple products for reactant set with index {i}"
                        )
                    elif error_level == "warn":
                        warnings.warn(
                            f"ERROR: Multiple products for reactant set with index {i}"
                        )
                    else:
                        raise ValueError(f"Unknown error level: {error_level}")

            mapped_reactions.append(reaction_inst_cleaned)

    return mapped_reactions


def remove_monomer_pg_chirality(
    mol: Union[str, Chem.Mol], no_match: str = "silent"
) -> Chem.Mol:
    """
    Given a monomer building block, remove only the chiral information
    for the spiro carbon of the protecting group, leaving the rest unchanged.

    Args:
        mol (Union[str, Chem.Mol]): The monomer given as SMILES string or RDKit Mol
        no_match (str): What to do if no substructure match is found.
            If "silent", just return the input mol, if "error", raise an exception.
            Default: "silent"

    Returns:
        Chem.Mol: The monomer with chiral information for the spiro carbon removed
    """
    if isinstance(mol, str):
        mol = Chem.MolFromSmiles(mol)
    else:
        mol = Chem.Mol(mol)

    patt = Chem.MolFromSmarts(
        "[$([CR2](O1)(ONC2)(C2)C(=O)OC1)]"
    )  # hits a spiro carbon atom between an isoxazolidine and a 5-membered lactone-ether ring

    match = mol.GetSubstructMatches(patt)
    if len(match) == 0:
        if no_match == "silent":
            return mol
    if len(match) != 1:
        raise ValueError(
            f"{len(match)} substructure matches found for {Chem.MolToSmiles(mol)}. Expected 1."
        )
    atom = mol.GetAtomWithIdx(match[0][0])
    atom.SetChiralTag(Chem.ChiralType.CHI_UNSPECIFIED)
    return mol

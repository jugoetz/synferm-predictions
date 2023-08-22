from rdkit import Chem
from rdkit.Chem import Draw, rdChemReactions
from rdkit.Chem.rdchem import Mol


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


def map_reactions(rxn, reactant_sets):
    """Take a reaction template and a list of reactant sets and return the mapped reactions."""
    ketone_slap_substructure = Chem.MolFromSmiles("NC(COC[Si](C)(C)C)(C)C")
    mapped_reactions = []
    for i, reactant_set in enumerate(reactant_sets):
        reaction_inst = create_reaction_instance(rxn, reactant_set)
        if len(reaction_inst) == 1:  # all good
            mapped_reactions.append(reaction_inst[0])
        elif len(reaction_inst) == 0:  # failed
            mapped_reactions.append(None)
            print(f"ERROR: No product for reactant set with index {i}")
        elif len(reaction_inst) == 2 and reaction_inst[0].GetReactants()[
            1
        ].HasSubstructMatch(
            ketone_slap_substructure
        ):  # it's a ketone so it will give two products
            # ketone SLAP products may or may not be identical, depending on whether the starting ketone was assymetric
            # to compare the smiles, we need to remove molAtomMapNumbers. We need to work on copies to not clear them from the reaction instance
            products = [
                Chem.Mol(reaction_inst[0].GetProducts()[0]),
                Chem.Mol(reaction_inst[1].GetProducts()[0]),
            ]
            for p in products:
                for atom in p.GetAtoms():
                    if atom.HasProp("molAtomMapNumber"):
                        atom.ClearProp("molAtomMapNumber")
            if Chem.MolToSmiles(products[0]) == Chem.MolToSmiles(
                products[1]
            ):  # products are identical and we can discard one reaction
                mapped_reactions.append(reaction_inst[0])
            else:
                print(
                    f"WARNING: Multiple stereoisomeric products for reactant set with index {i}"
                )
                mapped_reactions.append(reaction_inst)
        else:  # failed
            mapped_reactions.append(reaction_inst)
            print(f"ERROR: Multiple products for reactant set with index {i}")
    return mapped_reactions

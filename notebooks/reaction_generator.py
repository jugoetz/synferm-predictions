"""
code copied from library-generation project
"""

from typing import List, Union, Tuple, Sequence

from rdkit import Chem
from rdkit.Chem.rdChemReactions import (
    ChemicalReaction,
    ReactionFromSmarts,
    SanitizeRxn,
    ReactionToSmiles,
)

from src.util.rdkit_util import (
    map_reactions,
)


class SFReactionGenerator:
    # reactants to product A
    _rxn_abt = "[$(B(-F)(-F)-F)]-[$(C-[#6])X3:1]=[O:2].O=C1-O-[$(C2CCCCC2)]-O-[C:6]-1-1-[C:5]-[C:4]-[NH1:3]-O-1.[NH2:7]-[c:8]1:[c:9]:[c:10]:[c:11]:[c:12]:[c:13]:1-[SH1:14]>>[C:1](=[O:2])-[N:3]-[C:4]-[C:5]-[c:6]1:[n:7]:[c:8]2:[c:9]:[c:10]:[c:11]:[c:12]:[c:13]:2:[s:14]:1."
    _rxn_th = "[$(B(-F)(-F)-F)]-[$(C-[#6])X3:1]=[O:2].O=C1-O-[$(C2CCCCC2)]-O-[C:6]-1-1-[C:5]-[C:4]-[NH1:3]-O-1.[C:7](=[S:8])-[NH1:9]-[NH2:10]>>[c:7]1:[n:9]:[n:10]:[c:6](-[C:5]-[C:4]-[N:3]-[C:1]=[O:2]):[s:8]:1."

    # product A to reactants
    _backwards_rxn_abt = "[$(C-[#6]):1](=O)-[NR0]-[C:2]-[C:3]-c1nc2[c:4][c:5][c:6][c:7]c2s1>>F-[B-](-F)(-F)-[C:1]=O.O=C1-O-C2(-C-C-C-C-C-2)-O-C-1-1-[C:3]-[C:2]-N-O1.N-c1:[c:4]:[c:5]:[c:6]:[c:7]:c:1-S"
    _backwards_rxn_th = "[c:4]1nnc(-[C:3]-[C:2]-[NR0]-[$(C-[#6]):1]=O)s1>>F-[B-](-F)(-F)-[C:1]=O.O=C1-O-C2(-C-C-C-C-C-2)-O-C-1-1-[C:3]-[C:2]-N-O1.[C:4](=S)-N-N"

    def __init__(self):
        # Note: the reaction Sanitization warnings are due to the fact that the reaction templates contain dummy atoms
        # that are not mapped to the products (b/c they are not contained in the products ffs)
        # I don't get why RDKit does this as it is an intended and necessary use-case, but as usual their documentation
        # does not help with finding the reason.
        # Sample warnings:
        # Could not find RLabel mapping for atom: 0 in template: 0
        # Mismatched potential rlabels: 1 unmapped reactant dummy atom rlabels,
        # 0 unmappped (sic!) product dummy atom rlabels

        self.backwards_reactions = {
            "abt": ReactionFromSmarts(self._backwards_rxn_abt),
            "th": ReactionFromSmarts(self._backwards_rxn_th),
        }

        self.forward_reactions = {
            "abt": ReactionFromSmarts(self._rxn_abt),
            "th": ReactionFromSmarts(self._rxn_th),
        }

        for rxn in self.backwards_reactions.values():
            rxn.Initialize()
            SanitizeRxn(rxn)
        for rxn in self.forward_reactions.values():
            rxn.Initialize()
            SanitizeRxn(rxn)

    def _try_reaction(
        self, product_type: str, product_mol: Chem.Mol
    ) -> Union[Tuple[List[List[Chem.Mol]], str], Tuple[None, None]]:
        reactants = self.backwards_reactions[product_type].RunReactants((product_mol,))
        if len(reactants) > 0:
            # sanitize before returning
            [[Chem.SanitizeMol(m) for m in pair] for pair in reactants]
            return reactants, product_type
        return None, None

    def generate_reactants(self, product: Union[str, Chem.Mol]) -> List[Chem.Mol]:
        """
        Takes an SF reaction product (product A) and generates the reactants leading to this product.
        Note that the monomer will have an unspecified chiral center at the spiro carbon atom of the keto-acid
        protecting group. This chiral center is not transferred to the product.

        Args:
            product (str or Chem.Mol): Product of a Synthetic Fermentation reaction.

        Returns:
            list: A tuple of 3 reactants (as Mol objects) in the order initiator, monomer, terminator.

        Raises:
            ValueError: If the given product is not a valid Synthetic Fermentation reaction product.
            RuntimeError: If more than one possible reactions is found. This is a sanity check and should never happen.
                We include it because SMARTS pattern matching can sometimes give unforeseen results due to symmetry.
        """
        if isinstance(product, str):
            product_mol = Chem.MolFromSmiles(product)
        else:
            product_mol = Chem.Mol(product)

        # we try applying both reaction templates. We stop as soon as a template works
        # abt
        reactants, product_type = self._try_reaction("abt", product_mol)
        if not reactants:
            # th
            reactants, product_type = self._try_reaction("th", product_mol)
        if not reactants:
            raise ValueError(
                "The given product is not a valid Synthetic Fermentation reaction product."
            )

        # sanity check: there should never be more than one possible reaction
        if len(reactants) > 1:
            raise RuntimeError("More than one possible reaction found.")

        return reactants[0]

    def generate_product(self, reactants: List[Chem.Mol]) -> Chem.Mol:
        """
        Generates the product of an SF reaction given the reactants.

        Args:
            reactants (list): Reactants of the Synthetic Fermentation reaction.
                Expects an initiator, monomer, and terminator in this order.

        Returns:
            Chem.Mol: Product of the SF reaction.
        """

        prods = self.forward_reactions["abt"].RunReactants(reactants)
        if len(prods) == 0:
            prods = self.forward_reactions["th"].RunReactants(reactants)

        if len(prods) == 0:
            raise RuntimeError("No product found.")
        elif len(prods) > 1:
            raise RuntimeError("More than one product found.")

        return prods[0][0]

    def generate_reaction(
        self,
        reactants: Sequence[Chem.Mol],
    ) -> Union[ChemicalReaction, Tuple[ChemicalReaction, Chem.Mol]]:
        """
        Generates an atom-mapped, unbalanced reaction from the reactants and product type.
        Note that the monomer reactant will have an unspecified chiral center at the spiro carbon atom of the keto-acid.

        Args:
            reactants (Sequence): Reactants of the Synthetic Fermentation reaction.
                Expects an initiator, monomer, and terminator in this order.

        Returns:
            ChemicalReaction: A ChemicalReaction object representing the SF reaction.

        Raises:
            ValueError: If not exactly one reaction is found for the given reactants.
        """

        try:
            reaction = map_reactions(
                self.forward_reactions["abt"], (reactants,), error_level="error"
            )[
                0
            ]  # we expect exactly one reaction (or an exception)
        except RuntimeError as e1:
            # if this didn't work, try TH reaction template
            try:
                reaction = map_reactions(
                    self.forward_reactions["th"], (reactants,), error_level="error"
                )[
                    0
                ]  # we expect exactly one reaction (or an exception)
            except RuntimeError as e2:
                raise RuntimeError(
                    f"Encountered Errors on ABT and TH reaction templates while generating reaction for building blocks: "
                    f"'{Chem.MolToSmiles(reactants[0])}', "
                    f"'{Chem.MolToSmiles(reactants[1])}', "
                    f"and '{Chem.MolToSmiles(reactants[2])}'.\n"
                    f"Original error messages: {e1}\n{e2}"
                )

        return reaction

    def get_reaction_smiles(
        self,
        product: Union[str, Chem.Mol],
    ) -> str:
        """
        Generates the atom-mapped SF reactionSMILES leading to the given product.
        Note that the monomer reactant will have an unspecified chiral center at the spiro carbon atom of the keto-acid.
        This means that the reactionSMILES generated here may not be identical to the reactionSMILES in our
        "curated data", which was generated by a different method.
        Differences may also occur in the exact numbers used for atom-mapping, but not in the mapping itself.
        For ML, this is inconsequential, as the CGR generally invariant to the mapping numbers and, using our "custom"
        featurizer also invariant to chirality.

        Args:
            product (str or Chem.Mol): Product (type A) of an SF reaction.

        Returns:
            str: reactionSMILES leading to the given product.
        """
        reactants = self.generate_reactants(product)

        try:
            reaction = self.generate_reaction(reactants)

        except (RuntimeError, ValueError) as e:
            raise RuntimeError(
                f"Encountered Error while generating reaction for product '{product}'.\n"
                f"Original error message: {e}"
            )

        return ReactionToSmiles(reaction)

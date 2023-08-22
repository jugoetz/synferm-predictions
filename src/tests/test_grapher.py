from unittest import TestCase

import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem.rdChemReactions import ReactionFromSmarts

from src.data.grapher import (
    get_atom_to_bond_maps,
    create_cgr,
    build_mol_graph,
    get_atom_map_numbers,
    find_indices,
    atom_and_bond_mapping,
)
from src.data.featurizers import dummy_atom_featurizer, dummy_bond_featurizer
from src.util.rdkit_util import mol_to_file_with_indices


class Test(TestCase):
    def setUp(self) -> None:
        self.example_smi = "C1NC1C"  # 2-methylaziridine
        self.example_m = Chem.MolFromSmiles(self.example_smi)
        self.example_rxn_smiles = (
            "Br[Br:4].[CH2:0]1[NH:1][CH:2]1[CH3:3]>>[CH2:0]1[NH:1][CH:2]1[CH2:3][Br:4]"
        )
        self.example_rxn = ReactionFromSmarts(self.example_rxn_smiles)

        # extract reactant/product Mol (i.e. a single Mol for all reactants and one for all products)
        self.reactants = Chem.MolFromSmiles(self.example_rxn_smiles.split(">>")[0])
        self.products = Chem.MolFromSmiles(self.example_rxn_smiles.split(">>")[1])

        # produce images containing ground truth indices
        mol_to_file_with_indices(self.reactants, "reactants.png")
        mol_to_file_with_indices(self.products, "products.png")

        # extract atom map numbers
        self.reactant_atom_map_numbers = get_atom_map_numbers(self.reactants)
        self.product_atom_map_numbers = get_atom_map_numbers(self.products)

        # extract atom-to-bond relations
        self.reactant_a2b = get_atom_to_bond_maps(self.reactants)
        self.product_a2b = get_atom_to_bond_maps(self.products)

        # graphs
        self.reactant_graph = build_mol_graph(
            self.reactants, dummy_atom_featurizer, dummy_bond_featurizer
        )
        self.product_graph = build_mol_graph(
            self.products, dummy_atom_featurizer, dummy_bond_featurizer
        )

    def test_get_atom_to_bond_maps(self):
        """The ground truth can be assessed with self.example_m.GetBondBetweenAtoms(x, y).GetIdx()"""
        expected_a2b = {
            (0, 1): 0,  # C_sec -> N
            (1, 2): 1,  # N -> C_tert
            (2, 3): 2,  # C_tert -> methyl
            (2, 0): 3,  # C_tert -> C_sec
        }
        a2b = get_atom_to_bond_maps(self.example_m)
        self.assertEqual(expected_a2b, a2b)

    def test_create_cgr_reac_prod(self):
        """
        Create a CGR with the dummy featurizers which featurize every atom/bond with its index.
        Note that atoms/bonds that do not show up in e.g. products will receive feature "0.".
        """
        cgr = create_cgr(
            self.reactants,
            self.products,
            dummy_atom_featurizer,
            dummy_bond_featurizer,
            "reac_prod",
        )

        with self.subTest("Testing atom order/features"):
            self.assertTrue(
                torch.all(
                    torch.eq(
                        torch.Tensor(
                            [
                                [0.0, 0.0],
                                [1.0, 4.0],
                                [2.0, 0.0],
                                [3.0, 1.0],
                                [4.0, 2.0],
                                [5.0, 3.0],
                            ]
                        ),
                        cgr.ndata["x"],
                    )
                )
            )

        with self.subTest("Testing bond order/features"):
            self.assertTrue(
                torch.all(
                    torch.eq(
                        torch.Tensor(
                            [
                                [1.0, 0.0],  # CH2-NH bond
                                [1.0, 0.0],
                                [2.0, 1.0],  # CH-N bond
                                [2.0, 1.0],
                                [3.0, 2.0],  # acyclic C-C bond
                                [3.0, 2.0],
                                [4.0, 4.0],  # cyclic C-C bond
                                [4.0, 4.0],
                                [0.0, 0.0],  # lost bond Br-Br
                                [0.0, 0.0],
                                [0.0, 3.0],  # formed bond C-Br
                                [0.0, 3.0],
                            ]
                        ),
                        cgr.edata["e"],
                    )
                )
            )

    def test_create_cgr_reac_diff(self):
        """
        Analogous to test_create_cgr_reac_prod, but with different aggregation mode for features
        """
        cgr = create_cgr(
            self.reactants,
            self.products,
            dummy_atom_featurizer,
            dummy_bond_featurizer,
            "reac_diff",
        )

        with self.subTest("Testing atom order/features"):
            self.assertTrue(
                torch.all(
                    torch.eq(
                        torch.Tensor(
                            [
                                [0.0, 0.0],
                                [1.0, 3.0],
                                [2.0, -2.0],
                                [3.0, -2.0],
                                [4.0, -2.0],
                                [5.0, -2.0],
                            ]
                        ),
                        cgr.ndata["x"],
                    )
                )
            )

        with self.subTest("Testing bond order/features"):
            self.assertTrue(
                torch.all(
                    torch.eq(
                        torch.Tensor(
                            [
                                [1.0, -1.0],  # CH2-NH bond
                                [1.0, -1.0],
                                [2.0, -1.0],  # CH-N bond
                                [2.0, -1.0],
                                [3.0, -1.0],  # acyclic C-C bond
                                [3.0, -1.0],
                                [4.0, 0.0],  # cyclic C-C bond
                                [4.0, 0.0],
                                [0.0, 0.0],  # lost bond Br-Br
                                [0.0, 0.0],
                                [0.0, 3.0],  # formed bond C-Br
                                [0.0, 3.0],
                            ]
                        ),
                        cgr.edata["e"],
                    )
                )
            )

    def test_create_cgr_prod_diff(self):
        """
        Analogous to test_create_cgr_reac_prod, but with different aggregation mode for features
        """
        cgr = create_cgr(
            self.reactants,
            self.products,
            dummy_atom_featurizer,
            dummy_bond_featurizer,
            "prod_diff",
        )

        with self.subTest("Testing atom order/features"):
            self.assertTrue(
                torch.all(
                    torch.eq(
                        torch.Tensor(
                            [
                                [0.0, 0.0],
                                [4.0, -3.0],
                                [0.0, 2.0],
                                [1.0, 2.0],
                                [2.0, 2.0],
                                [3.0, 2.0],
                            ]
                        ),
                        cgr.ndata["x"],
                    )
                )
            )

        with self.subTest("Testing bond order/features"):
            self.assertTrue(
                torch.all(
                    torch.eq(
                        torch.Tensor(
                            [
                                [0.0, 1.0],  # CH2-NH bond
                                [0.0, 1.0],
                                [1.0, 1.0],  # CH-N bond
                                [1.0, 1.0],
                                [2.0, 1.0],  # acyclic C-C bond
                                [2.0, 1.0],
                                [4.0, 0.0],  # cyclic C-C bond
                                [4.0, 0.0],
                                [0.0, 0.0],  # lost bond Br-Br
                                [0.0, 0.0],
                                [3.0, -3.0],  # formed bond C-Br
                                [3.0, -3.0],
                            ]
                        ),
                        cgr.edata["e"],
                    )
                )
            )

    def test_find_indices(self):
        arr = np.array([1, 2, 3, 4, 5])
        search_values = np.array([5, 3, 2])
        index_arr = find_indices(arr, search_values)
        self.assertTrue((arr[index_arr] == search_values).all())

    def test_atom_and_bond_mapping_returns_correct_ronly_atom(self):
        """Ground truth: atom 0 from reactants (Br) does not end up in product"""
        _, ronly_atom, _, _, _, _ = atom_and_bond_mapping(
            self.reactant_atom_map_numbers,
            self.product_atom_map_numbers,
            self.reactant_a2b,
            self.product_a2b,
        )
        self.assertEqual(
            [
                0,
            ],
            ronly_atom,
        )

    def test_atom_and_bond_mapping_returns_correct_ponly_atom(self):
        """All atoms from the products are present in reactants, must return an empty list"""
        _, _, ponly_atom, _, _, _ = atom_and_bond_mapping(
            self.reactant_atom_map_numbers,
            self.product_atom_map_numbers,
            self.reactant_a2b,
            self.product_a2b,
        )
        self.assertFalse(ponly_atom)

    def test_atom_and_bond_mapping_returns_correct_r2p_atom(self):
        """
        See reactants.png and products.png for ground truth mapping.
        Mapping should be [(1, 4), (2, 0), (3, 1), (4, 2), (5, 3)]
        """
        r2p_atom, _, _, _, _, _ = atom_and_bond_mapping(
            self.reactant_atom_map_numbers,
            self.product_atom_map_numbers,
            self.reactant_a2b,
            self.product_a2b,
        )
        self.assertEqual([(1, 4), (2, 0), (3, 1), (4, 2), (5, 3)], r2p_atom)

    def test_atom_and_bond_mapping_returns_correct_ronly_bond(self):
        """
        Ground truth: Only the Br-Br bond is not in products.
        This is the bond between reactant atoms 0 and 1, so it has index 0.
        (note that opposed to the graphs, here we still have only one bond to represent each actual bond)
        """
        _, _, _, _, ronly_bond, _ = atom_and_bond_mapping(
            self.reactant_atom_map_numbers,
            self.product_atom_map_numbers,
            self.reactant_a2b,
            self.product_a2b,
        )
        self.assertEqual(
            [
                0,
            ],
            ronly_bond,
        )

    def test_atom_and_bond_mapping_returns_correct_ponly_bond(self):
        """
        Ground truth: Only the Br-C bond is not in reactants.
        This is the bond between atoms 3 and 4 in the products.
        (Mol.GetBondBetweenAtoms(3,4).GetIdx() reveals this is bond 3).
        """
        _, _, _, _, _, ponly_bond = atom_and_bond_mapping(
            self.reactant_atom_map_numbers,
            self.product_atom_map_numbers,
            self.reactant_a2b,
            self.product_a2b,
        )
        self.assertEqual(
            [
                3,
            ],
            ponly_bond,
        )

    def test_atom_and_bond_mapping_returns_correct_r2p_bond(self):
        """
        See reactants.png and products.png for ground truth mapping.
        You can use Mol.GetBondBetweenAtoms(start_atom, end_atom).GetIdx() to get the
        ground truth bond mapping from there, which is:
        [(1, 0), (2, 1), (3, 2), (4, 4)]
        """
        _, _, _, r2p_bond, _, _ = atom_and_bond_mapping(
            self.reactant_atom_map_numbers,
            self.product_atom_map_numbers,
            self.reactant_a2b,
            self.product_a2b,
        )
        self.assertEqual([(1, 0), (2, 1), (3, 2), (4, 4)], r2p_bond)

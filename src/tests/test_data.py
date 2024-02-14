import os
from unittest import TestCase

import numpy as np
import torch

from src.data.dataloader import SynFermDataset, GraphLessSynFermDataset
from src.data.featurizers import OneHotEncoder
from src.util.definitions import DATA_ROOT


class TestDataCaching(TestCase):
    def setUp(self) -> None:
        self.data = SynFermDataset(
            name="example_sf.csv",
            raw_dir=DATA_ROOT,
            save_dir=(DATA_ROOT / "cache"),
            reaction=True,
            smiles_columns=["reaction_smiles_atom_mapped"],
            label_columns=["binary_A"],
            graph_type="bond_edges",
            global_features=None,
            global_features_file=None,
            featurizers="custom",
            task="binary",
            force_reload=True,  # this will be cached freshly
        )

        self.cached_data = SynFermDataset(
            name="example_sf.csv",
            raw_dir=DATA_ROOT,
            save_dir=(DATA_ROOT / "cache"),
            reaction=True,
            smiles_columns=["reaction_smiles_atom_mapped"],
            label_columns=["binary_A"],
            graph_type="bond_edges",
            global_features=None,
            global_features_file=None,
            featurizers="custom",
            task="binary",
            force_reload=False,  # this will be loaded from cache
        )

        assert (
            self.data.hash == self.cached_data.hash
        )  # assert that hashes actually align

    def test_reload_gives_same_data_length(self):
        self.assertEqual(len(self.data.graphs), len(self.cached_data.graphs))

    def test_reload_gives_same_global_features(self):
        self.assertTrue(
            torch.all(self.data.global_features == self.cached_data.global_features)
        )


class TestOneHotEncoder(TestCase):
    def setUp(self):
        self.encoder = OneHotEncoder()
        self.smiles = ["C", "CC", "CCC", "CCCC"]
        self.encoder.add_dimension(self.smiles)
        self.encoder.add_dimension(self.smiles)
        self.encoder.add_dimension(self.smiles)

    def test_save_state_dict(self):
        """Test that a save/load roundtrip works"""
        self.encoder.save_state_dict("test_save_state_dict.json")
        encoder_from_file = OneHotEncoder()
        encoder_from_file.load_state_dict("test_save_state_dict.json")
        os.remove("test_save_state_dict.json")
        self.assertEqual(self.encoder.n_dimensions, encoder_from_file.n_dimensions)

    def test_process(self):
        for i, smi in enumerate(self.smiles):
            with self.subTest(smi=smi):
                ohe = self.encoder.process(smi, smi, smi)
                self.assertTrue(
                    np.all(
                        ohe.nonzero()
                        == np.array([i, i + len(self.smiles), i + 2 * len(self.smiles)])
                    )
                )

    def test_process_after_save_load(self):
        self.encoder.save_state_dict("test_process_after_save_load.json")
        encoder_from_file = OneHotEncoder()
        encoder_from_file.load_state_dict("test_process_after_save_load.json")
        os.remove("test_process_after_save_load.json")
        for i, smi in enumerate(self.smiles):
            with self.subTest(smi=smi):
                ohe = encoder_from_file.process(smi, smi, smi)
                self.assertTrue(
                    np.all(
                        ohe.nonzero()
                        == np.array([i, i + len(self.smiles), i + 2 * len(self.smiles)])
                    )
                )

    def test_unknown_molecule_raises_value_error(self):
        with self.assertRaises(ValueError):
            self.encoder.process(self.smiles[0], self.smiles[1], "c1ccccc1")

    def test_unknown_molecule_fails_silent(self):
        silent_encoder = OneHotEncoder(unknown_molecule="silent")
        silent_encoder.add_dimension(self.smiles)
        self.assertEqual(0, silent_encoder.process("c1ccccc1").sum())


class TestGraphLessSynFermDataset(TestCase):
    def setUp(self) -> None:
        self.data = GraphLessSynFermDataset(
            name="example_sf.csv",
            raw_dir=DATA_ROOT,
            global_features=["OHE_silent"],
            smiles_columns=["I_smiles", "M_smiles", "T_smiles"],
            label_columns=None,
            task="multilabel",
            force_reload=True,
        )

    def test_ohe_data_sizes_correct(self):
        # in the example data there are 1 I, 2 M, 8 T, so these are the sizes we expect
        correct = {"OHE_0": 1, "OHE_1": 2, "OHE_2": 8}
        for name, size in self.data.global_features_sizes:
            with self.subTest(name=name):
                self.assertEqual(correct[name], size)

    def test_ohe_known_encodings_correct(self):
        # we initialize the OHEncoder with this data, so all reactants must be known
        for i in range(len(self.data)):
            with self.subTest(idx=i):
                self.assertTrue(all(self.data.known_one_hot_encodings(i)))

    def test_ohe_unknown_encodings_correct(self):
        # we initialize the OHEncoder with this data, so all reactants must be known
        data_with_unknown = GraphLessSynFermDataset(
            name="example_sf.csv",
            raw_dir=DATA_ROOT,
            global_features=["OHE_silent"],
            smiles_columns=["I_smiles", "M_smiles", "T_smiles"],
            label_columns=None,
            task="multilabel",
            force_reload=True,
            global_featurizer_state_dict_path=(
                DATA_ROOT / "OHE_state_dict_example.json"
            ),  # a fabricated state dict that recognizes only Ph023,Mon017,TerTH010 from example_sf.csv
        )
        # the expected result is that the first element in the tuple (I) it always True
        # the second element (M) is true for indices 0-7 (inclusive)
        # the third element (T) is true for indices 0 and 8

        for i in range(len(data_with_unknown)):
            with self.subTest(idx=i):
                if i == 0:
                    self.assertListEqual(
                        [True, True, True],
                        list(data_with_unknown.known_one_hot_encodings(i)),
                    )
                elif i < 8:
                    self.assertListEqual(
                        [True, True, False],
                        list(data_with_unknown.known_one_hot_encodings(i)),
                    )
                elif i > 8:
                    self.assertListEqual(
                        [True, False, False],
                        list(data_with_unknown.known_one_hot_encodings(i)),
                    )
                else:
                    self.assertListEqual(
                        [True, False, True],
                        list(data_with_unknown.known_one_hot_encodings(i)),
                    )

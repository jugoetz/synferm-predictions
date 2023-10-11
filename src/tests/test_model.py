from unittest import TestCase

import dgl
import torch

from src.model.classifier import load_trained_model, AttentiveFPModel
from src.util.definitions import TRAINED_MODEL_DIR


class TestAttentiveFPModel(TestCase):
    def setUp(self):
        atom_feat_size = 6
        bond_feat_size = 10
        global_feat_size = 15
        self.batch_size = 2
        self.num_labels = 3
        edge_src = torch.tensor([2, 3, 4])  # src node ids
        edge_dest = torch.tensor([1, 2, 3])  # dest node ids
        graph = dgl.graph((edge_src, edge_dest))
        graph.ndata["x"] = torch.rand((5, atom_feat_size))
        graph.edata["e"] = torch.rand((3, bond_feat_size))

        self.graph = dgl.batch([graph for _ in range(self.batch_size)])
        self.model = AttentiveFPModel(
            atom_feature_size=atom_feat_size,
            bond_feature_size=bond_feat_size,
            global_feature_size=global_feat_size,
            num_labels=self.num_labels,
            encoder={
                "depth": 5,
                "hidden_size": 300,
                "dropout_ratio": 0.2,
                "aggregation": "sum",
            },
            decoder={"depth": 3, "hidden_size": 32, "dropout_ratio": 0.2},
            training={"task": "binary"},
        )
        self.global_features = torch.randint(0, 10, (self.batch_size, global_feat_size))

    def test_forward_method_returns_tensor(self):
        self.assertTrue(
            isinstance(self.model((self.graph, self.global_features)), torch.Tensor)
        )

    def test_forward_method_returns_of_correct_shape(self):
        # shape should be batch_size x n_labels
        self.assertEqual(
            self.model((self.graph, self.global_features)).shape,
            (self.batch_size, self.num_labels),
        )


class TestTrainedModelLoading(TestCase):
    def setUp(self):
        # TODO paths are outdated
        self.ffn = load_trained_model(
            "FFN", TRAINED_MODEL_DIR / "2022-12-16-144509_863758" / "best.ckpt"
        )
        self.ffn.eval()
        self.dmpnn1d = load_trained_model(
            "D-MPNN", TRAINED_MODEL_DIR / "2022-12-16-145840_448790" / "best.ckpt"
        )
        self.dmpnn1d.eval()

    def test_ffn_first_linear_layer_has_correct_input_size(self):
        self.assertEqual(self.ffn.decoder._modules["ffn"][1].in_features, 132)

    def test_ffn_second_linear_layer_has_correct_hidden_size(self):
        self.assertEqual(self.ffn.decoder._modules["ffn"][4].in_features, 59)

    def test_ffn_dropout_has_correct_value(self):
        self.assertAlmostEqual(
            self.ffn.decoder._modules["ffn"][3].p, 3.67e-05, places=7
        )

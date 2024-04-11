from unittest import TestCase

import dgl
import torch

from src.model.classifier import AttentiveFPModel


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

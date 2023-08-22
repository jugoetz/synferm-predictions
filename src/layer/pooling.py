"""
Pooling layers for heterogeneous graphs

These are generalizations of the pooling layers in dgl.nn.pytorch.glob
"""

import torch
import torch.nn as nn
import numpy as np
import dgl


class AvgPooling(nn.Module):
    """Apply average pooling over the nodes in a graph.

    .. math::
        r^{(i)} = \frac{1}{N_i}\sum_{k=1}^{N_i} x^{(i)}_k

    Notes:
        Input: Could be one graph, or a batch of graphs. If using a batch of graphs,
        make sure nodes in all graphs have the same feature size, and concatenate
        nodes' feature together as the input.
    """

    def __init__(self, ntype, feat):
        """
        Args:
            ntype (str): Node type which features should be taken from
            feat (str): Feature that pooling is applied to
        """
        super(AvgPooling, self).__init__()
        self.ntype = ntype
        self.feat = feat

    def forward(self, graph):
        """
        Compute average pooling.

        Args:
            graph(DGLHeteroGraph): A DGLHeteroGraph or a batch of DGLHeteroGraphs.

        Returns:
            torch.Tensor: The output feature with shape :math:`(B, D)`, where
                :math:`B` refers to the batch size of input graphs.
        """
        readout = dgl.readout_nodes(graph, self.feat, ntype=self.ntype, op="mean")
        return readout


class SumPooling(nn.Module):
    """Apply sum pooling over the nodes in a graph.

    .. math::
        r^{(i)} = \sum_{k=1}^{N_i} x^{(i)}_k

    Notes:
        Input: Could be one graph, or a batch of graphs. If using a batch of graphs,
        make sure nodes in all graphs have the same feature size, and concatenate
        nodes' feature together as the input.
    """

    def __init__(self, ntype, feat):
        """
        Args:
            ntype (str): Node type which features should be taken from
            feat (str): Feature that pooling is applied to
        """
        super(SumPooling, self).__init__()
        self.ntype = ntype
        self.feat = feat

    def forward(self, graph):
        """
        Compute sum pooling.

        Args:
            graph(DGLHeteroGraph): A DGLHeteroGraph or a batch of DGLHeteroGraphs.

        Returns:
            torch.Tensor: The output feature with shape :math:`(B, D)`, where
                :math:`B` refers to the batch size of input graphs.
        """
        readout = dgl.readout_nodes(graph, self.feat, ntype=self.ntype, op="sum")
        return readout


class MaxPooling(nn.Module):
    """Apply max pooling over the nodes in a graph.

    .. math::
        r^{(i)} = \max_{k=1}^{N_i}\left( x^{(i)}_k \right)

    Notes:
        Input: Could be one graph, or a batch of graphs. If using a batch of graphs,
        make sure nodes in all graphs have the same feature size, and concatenate
        nodes' feature together as the input.
    """

    def __init__(self, ntype, feat):
        """
        Args:
            ntype (str): Node type which features should be taken from
            feat (str): Feature that pooling is applied to
        """
        super(MaxPooling, self).__init__()
        self.ntype = ntype
        self.feat = feat

    def forward(self, graph):
        """
        Compute max pooling.

        Args:
            graph(DGLHeteroGraph): A DGLHeteroGraph or a batch of DGLHeteroGraphs.

        Returns:
            torch.Tensor: The output feature with shape :math:`(B, D)`, where
                :math:`B` refers to the batch size of input graphs.
        """
        readout = dgl.readout_nodes(graph, self.feat, ntype=self.ntype, op="max")
        return readout


class GlobalAttentionPooling(nn.Module):
    """Global Attention Pooling from `Gated Graph Sequence Neural Networks
    <https://arxiv.org/abs/1511.05493>`__

    .. math::
        r^{(i)} = \sum_{k=1}^{N_i}\mathrm{softmax}\left(f_{gate}
        \left(x^{(i)}_k\right)\right) f_{feat}\left(x^{(i)}_k\right)

    Args:
        gate_nn (torch.nn.Module): A neural network that computes attention scores for each feature.
        ntype (str): Node type which features should be taken from
        feat (str): Feature that pooling is applied to
        feat_nn (torch.nn.Module, optional): A neural network applied to each feature before combining them with
            attention scores.
    """

    def __init__(self, gate_nn, ntype, feat, feat_nn=None, get_attention=False):
        super(GlobalAttentionPooling, self).__init__()
        self.gate_nn = gate_nn
        self.ntype = ntype
        self.feat = feat
        self.feat_nn = feat_nn
        self.get_attention = get_attention

    def forward(self, graph):
        """
        Compute global attention pooling.

        Args:
            graph(DGLHeteroGraph): A DGLHeteroGraph or a batch of DGLHeteroGraphs.

        Returns:
            torch.Tensor: The output feature with shape :math:`(B, D)`, where
                :math:`B` refers to the batch size of input graphs.
            torch.Tensor (optional): The attention values of shape :math:`(N, 1)`, where :math:`N` is the number of
                nodes in the graph. This is returned only when :attr:`get_attention` is ``True`` when instantiating.
        """
        with graph.local_scope():
            gate = self.gate_nn(graph.nodes[self.ntype].data[self.feat])
            assert (
                gate.shape[-1] == 1
            ), "The output of gate_nn should have size 1 at the last axis."
            feat = (
                self.feat_nn(graph.nodes[self.ntype].data[self.feat])
                if self.feat_nn
                else graph.nodes[self.ntype].data[self.feat]
            )

            graph.nodes[self.ntype].data["gate"] = gate
            gate = dgl.softmax_nodes(graph, feat="gate", ntype=self.ntype)
            graph.nodes[self.ntype].data.pop("gate")

            graph.nodes[self.ntype].data["r"] = feat * gate
            readout = dgl.readout_nodes(graph, "r", ntype=self.ntype, op="sum")
            graph.nodes[self.ntype].data.pop("r")

            if self.get_attention:
                return readout, gate
            else:
                return readout


class ConcatenateNodeEdgeSumPooling(nn.Module):
    """Apply sum pooling over the nodes and separately over the edges in a graph. Concatenate the results.

    Notes:
        Input: Could be one graph, or a batch of graphs. If using a batch of graphs,
        make sure nodes/edges in all graphs have the same feature size, and concatenate
        nodes'/edges' feature together as the input.
    """

    def __init__(self, ntype, etype, nfeat, efeat):
        """
        Args:
            ntype (str): Node type which features should be taken from
            feat (str): Feature that pooling is applied to
        """
        super(ConcatenateNodeEdgeSumPooling, self).__init__()
        self.ntype = ntype
        self.nfeat = nfeat
        self.etype = etype
        self.efeat = efeat

    def forward(self, graph):
        """
        Compute sum pooling.

        Args:
            graph(DGLHeteroGraph): A DGLHeteroGraph or a batch of DGLHeteroGraphs.

        Returns:
            torch.Tensor: The output feature with shape :math:`(B, D)`, where
                :math:`B` refers to the batch size of input graphs.
        """
        nreadout = dgl.readout_nodes(graph, self.nfeat, ntype=self.ntype, op="sum")
        ereadout = dgl.readout_edges(graph, self.efeat, etype=self.etype, op="sum")
        return torch.cat([nreadout, ereadout], dim=1)

from typing import Union

import dgl
import dgl.function as fn
import torch
import torch.nn as nn

from src.layer.util import get_activation
from src.layer.pooling import AvgPooling, SumPooling, MaxPooling, GlobalAttentionPooling


class MPNNEncoder(nn.Module):
    """
    A :class:`MPNNEncoder` is a message passing neural network for encoding a molecule.
    """

    def __init__(
        self,
        atom_feature_size: int,
        bond_feature_size: int,
        *,
        hidden_size: int = 300,
        bias: bool = False,
        depth: int = 3,
        dropout_ratio: float = 0.0,
        aggregation: str = "mean",
        activation: Union[str, callable] = "ReLU",
        **kwargs,
    ):
        """
        Several functions from Chemprop are not available as of now (b/c we don't need them):
            - atom_messages
            - atom_descriptors
            - aggregation by norm

        Args:
            atom_feature_size (int): Atom feature vector size.
            bond_feature_size (int): Bond feature vector size.
            hidden_size (list): Dimensionality of hidden layers in MPNN.
            bias (bool): Whether to add bias to linear layers (except last layer).
            depth (int): Number of message passing steps. Includes in and out step. Must be >= 3.
            dropout_ratio (float): Dropout probability.
            aggregation (Literal['mean', 'sum', 'norm']): Aggregation scheme for atomic vectors into molecular vectors.
            activation (str, callable): Activation function. If a string is passed, it must correspond to the name
                of a class in torch.nn.

        """
        super(MPNNEncoder, self).__init__()
        self.atom_fdim = atom_feature_size
        self.bond_fdim = bond_feature_size
        self.hidden_size = hidden_size
        self.bias = bias
        if depth < 3:
            raise ValueError(f"MPNN depth must be >= 3. Your choice {depth} is < 3.")
        self.depth = depth
        self.dropout = dropout_ratio
        self.aggregation = aggregation

        # Dropout
        self.dropout_layer = nn.Dropout(p=self.dropout)

        # Activation
        self.act_func = get_activation(activation)

        # Input
        input_dim = self.atom_fdim + self.bond_fdim
        self.W_i = nn.Linear(input_dim, self.hidden_size, bias=self.bias)

        # Shared weight matrix across depths (default)
        self.W_h = nn.Linear(self.hidden_size, self.hidden_size, bias=self.bias)

        # Output
        self.W_o = nn.Linear(self.atom_fdim + self.hidden_size, self.hidden_size)

        # Pooling
        if self.aggregation == "attention":
            self.pooling = GlobalAttentionPooling(
                gate_nn=nn.Linear(self.hidden_size, 1),
                ntype="atom",
                feat="h_v",
                get_attention=True,
            )
        elif self.aggregation == "mean":
            self.pooling = AvgPooling(ntype="atom", feat="h_v")
        elif self.aggregation == "sum":
            self.pooling = SumPooling(ntype="atom", feat="h_v")
        elif self.aggregation == "max":
            self.pooling = MaxPooling(ntype="atom", feat="h_v")
        else:
            raise ValueError(
                "Aggregation must be one of ['max', 'mean', 'sum', 'attention']"
            )

    def forward(
        self,
        graph: dgl.DGLGraph,
    ) -> torch.FloatTensor:
        """
        Encodes CGR graphs (can be a batch of graphs, i.e. one graph containing multiple unconnected entities)

        In Yang et al. (2019), the D-MPNN is definied by these steps:
            1. Instantiate
                - For edge vw, concatenate source node features x_v and edge features e_vw
                - Multiply with learnable Matrix of shape (hidden_size x input_size)
                - Activate with ReLU
                to give the initial hidden state of an edge
            2. Message function (repeat for n_steps)
                - Message m_vw at step t+1 is the sum over hidden states at step t for all edge directed at v except
                 the reverse edge wv
                - The new hidden state h_vw at t+1 is the message m_vw at this step multiplied with another learnable
                 matrix W_m of shape (hidden_size x hidden_size), plus the initial hidden state for the edge. W_m is
                 shared for all message passing steps.
                 - Activate with ReLU
            3. Reduce function (at last step)
                - Generate node message m_v by summing over hidden states (at last time step) of all outgoing edges
                (this seems weird, but it is what they write)
                - Generate node hidden state h_v by concatenating node features x_v and message m_v, multiply with
                 a learnable matrix
                 - Activate with ReLU
            4. Readout / Pooling
                - Sum (or other pooling operation) over all node hidden states to obtain the hidden state for a molecule

        The focus on edges instead of nodes does however not integrate well with the message-passing framework,
        making chemprop's forward() code hard to read or modify. Clearly, if the edges (i.e. the bonds in the chemical sense)
        pass messages to each other, one would want the bonds to be nodes!
        --> We use graphs with atom and bond edges, see function `build_cgr()` in `grapher_cgr.py`

        Args:
            graph (dgl.DGLHeterograph): Single or batch of CGRs with atom and bond nodes.

        Returns:
            torch.Tensor: A tensor of shape (num_molecules, hidden_size) containing the encoding of each molecule.
        """

        # 1. Set initial states h_0
        #       we send atom features x via the ("atom", "starts", "bond") edges,
        #       reduce by sum (meaningless as every node gets only one message),
        #       update by concat, weight, activation
        def get_initial_hidden(nodes):
            """
            Here we conduct h_b = tau(W_i * concat(x, e))
            Where x is the incoming message, e the bond feature, W_i a linear layer and tau an activation.
            """
            concat_feats = torch.cat(
                (nodes.data["x"], nodes.data["e"]), dim=1
            )  # num_bond_nodes x (len(x) + len(e))
            out = self.act_func(self.W_i(concat_feats))  # num_bond_nodes x hidden_size
            return {"h_0": out}

        graph.update_all(
            fn.copy_u("x", "m_x"),
            fn.sum("m_x", "x"),
            apply_node_func=get_initial_hidden,
            etype=("atom", "starts", "bond"),
        )

        # 2. Main message passing phase:
        #       send bond node hidden state along ("bond", "leads_to", "bond") edges,
        #       reduce by sum,
        #       update by linear weight layer, then sum with node h_0, then activation

        def get_next_hidden(nodes):
            """
            Here we conduct h_b(t+1) = tau(h_0 + W_h * m)
            Where m is the sum of incoming messages m_b, W_h a linear layer, h_0 the initial bond hidden state and tau an activation.
            """
            out = self.act_func(
                nodes.data["h_0"] + self.W_h(nodes.data["m"])
            )  # num_nodes x hidden_size
            return {"h_b": out}

        # in the first layer, we start from h_0
        graph.update_all(
            fn.copy_u("h_0", "m_b"),
            fn.sum("m_b", "m"),
            apply_node_func=get_next_hidden,
            etype=("bond", "leads_to", "bond"),
        )

        # from the second layer onwards, the message is state h_b
        for _ in range(
            self.depth - 3
        ):  # -3 for input layer + separately treated 1st layer above + output layer
            graph.update_all(
                fn.copy_u("h_b", "m_b"),
                fn.sum("m_b", "m"),
                apply_node_func=get_next_hidden,
                etype=("bond", "leads_to", "bond"),
            )

        # 3. Condensation to atom states:
        #       As the last message passing step, we condense the bond hidden states to atom hidden states.
        #       Send the bond hidden state h_b along the ("bond", "starts_at", "atom") edges ,
        #       reduce by sum,
        #       update by concat(x, m), then linear layer, then activation.

        def get_atom_hidden(nodes):
            """
            Here we conduct h_v = tau(W_o * concat(x, m))
            Where x is the atom features, m the sum of bond messages m_b, W_o a linear layer and tau an activation.
            """
            concat_feats = torch.concat(
                (nodes.data["x"], nodes.data["m"]), dim=1
            )  # num_nodes x (len(x) + hidden_size)
            out = self.act_func(self.W_o(concat_feats))  # num_atom_nodes x hidden_size

            return {"h_v": out}

        graph.update_all(
            fn.copy_u("h_b", "m_b"),
            fn.sum("m_b", "m"),
            apply_node_func=get_atom_hidden,
            etype=("bond", "starts_at", "atom"),
        )

        # 4. Readout
        # We want to aggregate features over the nodes that belong to one graph.
        # So our output is of size (n_batched x n_features)

        if self.aggregation == "attention":
            hidden_rep, attention = self.pooling(graph)
        else:
            hidden_rep = self.pooling(graph)

        return hidden_rep  # n_batched x hidden_size

#!/usr/bin/env python3


import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


##########################################################
#                   DiffusionGraphConv
##########################################################

class DiffusionGraphConv(nn.Module):
    """
    Diffusion graph convolution operator with Chebyshev-like polynomial.
    Optionally: 'laplacian' => Chebyshev recursion,
                'dual_random_walk' => 2 directed RW supports, etc.
    """

    def __init__(
            self,
            num_supports: int,
            input_dim: int,
            hidden_dim: int,
            num_nodes: int,
            max_diffusion_step: int,
            output_dim: int,
            bias_start: float = 0.0,
            filter_type: str = 'laplacian',
    ):
        """
        Args:
            num_supports: number of support matrices (1 for 'laplacian',
                          2 for 'dual_random_walk')
            input_dim: input feature dimension (per node)
            hidden_dim: dimension of hidden state (per node)
            num_nodes: number of nodes in the graph
            max_diffusion_step: K (the order of the Chebyshev or # of hops)
            output_dim: dimension of the output (per node)
            bias_start: initial bias
            filter_type: 'laplacian' or 'dual_random_walk'
        """
        super().__init__()
        self.num_supports = num_supports
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_nodes = num_nodes
        self.max_diffusion_step = max_diffusion_step
        self.filter_type = filter_type

        # We do a conv on the concatenation of (X, H) => input_dim + hidden_dim
        # Then for each support and each diffusion step, we get another copy of the features.
        self._input_size = self.input_dim + self.hidden_dim

        # total # of “transformed” feature copies
        num_matrices = self.num_supports * self.max_diffusion_step + 1

        # Define weight and bias
        self.weight = nn.Parameter(
            torch.FloatTensor(self._input_size * num_matrices, output_dim)
        )
        self.bias = nn.Parameter(torch.FloatTensor(output_dim))

        # Initialize
        nn.init.xavier_normal_(self.weight, gain=1.414)
        nn.init.constant_(self.bias, bias_start)

    def forward(self, inputs_and_state: torch.Tensor, support_list: list):
        """
        inputs_and_state: shape (B, num_nodes, input_dim + hidden_dim)
        support_list: a list of adjacency-like matrices (PyTorch Tensors)
                      either of shape (num_nodes, num_nodes) or (B, num_nodes, num_nodes)
                      depending on whether it is dynamic or static adjacency.
                      Typically (num_nodes, num_nodes) for static graphs.
        Returns:
            Tensor of shape (B, num_nodes, output_dim)
        """
        b, n, in_feats = inputs_and_state.shape
        assert n == self.num_nodes
        assert in_feats == self._input_size

        # x0: (B, num_nodes, input_size)
        x0 = inputs_and_state
        x_stack = [x0]  # identity (0th order)

        # For each support in the list, apply Chebyshev recursion
        if self.max_diffusion_step > 0:
            for support in support_list:
                # x1 = support * x0  ( shape: (B, num_nodes, input_size) )
                # If support is static [n, n], then we do:
                x1 = torch.einsum('ij,bjk->bik', support, x0)
                # Add x1 to the stack
                x_stack.append(x1)
                # Keep track of x1,x0 for Chebyshev recursion
                x_prev, x_curr = x0, x1
                # Recursion for k=2..K
                for k in range(2, self.max_diffusion_step + 1):
                    x2 = 2 * torch.einsum('ij,bjk->bik', support, x_curr) - x_prev
                    x_stack.append(x2)
                    x_prev, x_curr = x_curr, x2

        # Now we have 1 + num_supports * max_diffusion_step “feature sets”
        # Concatenate them along the last dimension
        # x_stack => list of (B, num_nodes, input_size) with length “num_matrices”
        x_cat = torch.cat(x_stack, dim=-1)  # shape: (B, num_nodes, input_size * num_matrices)

        # Perform the linear transform
        # -> (B*num_nodes, input_size * num_matrices) @ (input_size * num_matrices, out_dim)
        b_n = b * n
        x_reshape = x_cat.view(b_n, -1)  # flatten
        out = x_reshape @ self.weight + self.bias  # => (B*n, out_dim)
        out = out.view(b, n, -1)  # => (B, num_nodes, out_dim)
        return out


##########################################################
#                   DCGRUCell
##########################################################

class DCGRUCell(nn.Module):
    """
    Diffusion Convolutional Gated Recurrent Unit Cell
    that uses DiffusionGraphConv for the gates.
    """

    def __init__(
            self,
            input_dim: int,
            num_units: int,
            max_diffusion_step: int,
            num_nodes: int,
            filter_type: str = "laplacian",
            nonlinearity: str = "tanh",
            use_gc_for_ru: bool = True,
    ):
        """
        Args:
            input_dim: Input feature dimension (per node)
            num_units: # of DCGRU hidden units (per node)
            max_diffusion_step: K (max Chebyshev order)
            num_nodes: # of nodes in the graph
            filter_type: 'laplacian' or 'dual_random_walk'
            nonlinearity: 'tanh' or 'relu' for candidate activation
            use_gc_for_ru: If True, use graph convolution for R/U gates
                           else uses a dense linear transform.
        """
        super().__init__()
        self.num_nodes = num_nodes
        self.num_units = num_units
        self.max_diffusion_step = max_diffusion_step
        self.use_gc_for_ru = use_gc_for_ru

        if nonlinearity == "tanh":
            self.act = torch.tanh
        elif nonlinearity == "relu":
            self.act = F.relu
        else:
            raise ValueError("Unknown nonlinearity.")

        # Determine how many supports
        if filter_type == "laplacian":
            self.num_supports = 1
        elif filter_type == "random_walk":
            self.num_supports = 1
        elif filter_type == "dual_random_walk":
            self.num_supports = 2
        else:
            self.num_supports = 1

        # Gate Convolution: output 2 * num_units for [r, u]
        if self.use_gc_for_ru:
            self.gate_conv = DiffusionGraphConv(
                num_supports=self.num_supports,
                input_dim=input_dim,
                hidden_dim=num_units,
                num_nodes=num_nodes,
                max_diffusion_step=max_diffusion_step,
                output_dim=num_units * 2,
                filter_type=filter_type,
                bias_start=1.0,  # per original code
            )
        else:
            # If not using graph conv for RU gates, use a dense layer
            in_channels = (input_dim + num_units)
            self.fc_gate = nn.Linear(in_channels, 2 * num_units, bias=True)
            nn.init.constant_(self.fc_gate.bias, 1.0)

        # Candidate Convolution
        self.candidate_conv = DiffusionGraphConv(
            num_supports=self.num_supports,
            input_dim=input_dim,
            hidden_dim=num_units,
            num_nodes=num_nodes,
            max_diffusion_step=max_diffusion_step,
            output_dim=num_units,
            filter_type=filter_type,
        )

    def forward(self, x: torch.Tensor, h: torch.Tensor, support_list: list):
        """
        Single time-step forward.
        Args:
            x: (B, num_nodes, input_dim)
            h: (B, num_nodes, num_units)  -- hidden state
            support_list: list of adjacency-like Tensors, shape (num_nodes, num_nodes)
                          or (B, num_nodes, num_nodes) for each support
        Returns:
            h_new: (B, num_nodes, num_units)
        """
        # r,u Gate
        if self.use_gc_for_ru:
            # Combine x,h => shape (B, num_nodes, in_dim + num_units)
            in_and_h = torch.cat([x, h], dim=-1)
            gates = self.gate_conv(in_and_h, support_list)  # (B, num_nodes, 2*num_units)
        else:
            # Dense version
            in_and_h = torch.cat([x, h], dim=-1)  # (B, num_nodes, in_dim+num_units)
            # Flatten last dim and do fc -> reshape
            B, N, _ = in_and_h.shape
            gates = self.fc_gate(in_and_h.view(B * N, -1))
            gates = gates.view(B, N, -1)  # (B, num_nodes, 2*num_units)

        r, u = torch.split(gates, self.num_units, dim=-1)
        r, u = torch.sigmoid(r), torch.sigmoid(u)

        # Candidate
        candidate_in = torch.cat([x, r * h], dim=-1)
        c = self.candidate_conv(candidate_in, support_list)  # (B, num_nodes, num_units)
        c = self.act(c)

        # New hidden state
        h_new = u * h + (1.0 - u) * c
        return h_new

    @property
    def output_size(self):
        return self.num_nodes * self.num_units

    def init_hidden(self, batch_size: int) -> torch.Tensor:
        """Return zeros for initial hidden state."""
        return torch.zeros(batch_size, self.num_nodes, self.num_units, dtype=torch.float32)


##########################################################
#                     TESTING
##########################################################

def main_test():
    # Suppose we have a directed graph with 5 nodes:
    num_nodes = 5
    # Build an adjacency (random for demonstration).
    np.random.seed(42)
    adj = np.random.rand(num_nodes, num_nodes)
    adj[adj < 0.7] = 0.0

    # Create 'dual_random_walk' support:
    # e.g., forward random walk, backward random walk
    def calc_rw(adj_mx):
        d = adj_mx.sum(axis=1)
        d_inv = np.where(d == 0, 0, 1 / d)
        return (np.diag(d_inv) @ adj_mx).astype(np.float32)

    forward_rw = torch.from_numpy(calc_rw(adj)).float()
    backward_rw = torch.from_numpy(calc_rw(adj.T)).float()

    support_list = [forward_rw, backward_rw]  # shape (5,5) each

    # Build a DCGRUCell
    cell = DCGRUCell(
        input_dim=16,
        num_units=8,
        max_diffusion_step=2,
        num_nodes=num_nodes,
        filter_type='dual_random_walk',
        nonlinearity='tanh',
        use_gc_for_ru=True,
    )

    # Batch of 4, each with (num_nodes=5, input_dim=16)
    x = torch.randn(4, num_nodes, 16)
    h = cell.init_hidden(batch_size=4)

    h_next = cell(x, h, support_list)  # single step
    print("Input shape: ", x.shape)
    print("Hidden shape (prev): ", h.shape)
    print("Hidden shape (new):  ", h_next.shape)


def generate_synthetic_eeg_data(batch_size, num_nodes, input_dim, seq_len):
    """
    Generate synthetic EEG data with normal and seizure-like activity.
    Args:
        batch_size: Number of samples in the batch.
        num_nodes: Number of electrodes (nodes in the graph).
        input_dim: Feature dimension per node (e.g., frequency bands).
        seq_len: Length of the time series.
    Returns:
        Tensor of shape (batch_size, seq_len, num_nodes, input_dim)
    """
    data = []
    for _ in range(batch_size):
        sample = []
        for t in range(seq_len):
            if t < seq_len // 2:  # Normal activity
                signal = np.random.normal(loc=0, scale=0.1, size=(num_nodes, input_dim))
            else:  # Seizure activity
                signal = np.random.normal(loc=1, scale=0.5, size=(num_nodes, input_dim))
            sample.append(signal)
        data.append(np.array(sample))
    return torch.tensor(np.array(data), dtype=torch.float32)


def create_graph_adjacency(num_nodes):
    """
    Create a random adjacency matrix for the graph.
    Args:
        num_nodes: Number of nodes in the graph.
    Returns:
        Adjacency matrix (torch.Tensor) of shape (num_nodes, num_nodes).
    """
    adj = np.random.rand(num_nodes, num_nodes)
    adj[adj < 0.7] = 0.0  # Sparse graph
    np.fill_diagonal(adj, 0)  # No self-loops
    return torch.tensor(adj, dtype=torch.float32)


def calc_rw(adj_mx):
    """
    Compute random walk transition matrix from adjacency matrix.
    """
    d = adj_mx.sum(dim=1)  # Sum along rows (degree of each node)
    d_inv = torch.where(d == 0, torch.tensor(0.0), 1.0 / d)  # Avoid division by zero
    # Compute D^-1 @ A (random walk transition matrix)
    return torch.diag(d_inv) @ adj_mx


def main_test_seizure_detection():
    # Parameters
    batch_size = 4
    num_nodes = 5  # Number of electrodes
    input_dim = 16  # Feature dimension (e.g., frequency bands)
    hidden_dim = 8  # Hidden state dimension
    seq_len = 20  # Time steps
    max_diffusion_step = 2
    filter_type = "dual_random_walk"

    # Generate synthetic EEG data
    eeg_data = generate_synthetic_eeg_data(batch_size, num_nodes, input_dim, seq_len)
    print("EEG Data Shape:", eeg_data.shape)  # (batch_size, seq_len, num_nodes, input_dim)

    # Create graph adjacency matrix and support list
    adj = create_graph_adjacency(num_nodes)
    forward_rw = calc_rw(adj)
    backward_rw = calc_rw(adj.T)
    support_list = [forward_rw, backward_rw]

    # Build DCGRUCell
    cell = DCGRUCell(
        input_dim=input_dim,
        num_units=hidden_dim,
        max_diffusion_step=max_diffusion_step,
        num_nodes=num_nodes,
        filter_type=filter_type,
        nonlinearity="tanh",
        use_gc_for_ru=True,
    )

    # Initialize hidden state
    h = cell.init_hidden(batch_size=batch_size)

    # Process each time step
    outputs = []
    for t in range(seq_len):
        x_t = eeg_data[:, t, :, :]  # Shape: (batch_size, num_nodes, input_dim)
        h = cell(x_t, h, support_list)
        outputs.append(h)

    # Stack outputs over time
    outputs = torch.stack(outputs, dim=1)  # Shape: (batch_size, seq_len, num_nodes, hidden_dim)
    print("Output Shape:", outputs.shape)

    # Example: Predict seizure likelihood
    classifier = nn.Linear(hidden_dim, 1)  # Binary classification (seizure vs normal)
    predictions = torch.sigmoid(classifier(outputs))  # Shape: (batch_size, seq_len, num_nodes, 1)
    print("Predictions Shape:", predictions.shape)

    # Print example predictions for the first sample
    print("Example Predictions (First Sample):")
    print(predictions[0, :, :, 0].detach().numpy())


if __name__ == "__main__":
    main_test_seizure_detection()

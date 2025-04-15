import numpy as np
import utils
import torch
import torch.nn as nn
from scipy.sparse import coo_matrix

class DiffusionGraphConv(nn.Module):
    def __init__(self, num_supports, input_dim, hid_dim, num_nodes,
                 max_diffusion_step, output_dim, bias_start=0.0,
                 filter_type='laplacian'):
        """
        Diffusion graph convolution
        Args:
            num_supports: number of supports, 1 for 'laplacian' filter and 2
                for 'dual_random_walk'
            input_dim: input feature dim
            hid_dim: hidden units
            num_nodes: number of nodes in graph
            max_diffusion_step: maximum diffusion step
            output_dim: output feature dim
            filter_type: 'laplacian' for undirected graph, and 'dual_random_walk'
                for directed graph
        """
        super(DiffusionGraphConv, self).__init__()
        num_matrices = num_supports * max_diffusion_step + 1
        self._input_size = input_dim + hid_dim
        self._num_nodes = num_nodes
        self._max_diffusion_step = max_diffusion_step
        self._filter_type = filter_type
        self.weight = nn.Parameter(
            torch.FloatTensor(
                size=(
                    self._input_size *
                    num_matrices,
                    output_dim)))
        self.biases = nn.Parameter(torch.FloatTensor(size=(output_dim,)))
        nn.init.xavier_normal_(self.weight.data, gain=1.414)
        nn.init.constant_(self.biases.data, val=bias_start)

    @staticmethod
    def _concat(x, x_):
        x_ = torch.unsqueeze(x_, 1)
        return torch.cat([x, x_], dim=1)

    @staticmethod
    def _build_sparse_matrix(L):
        """
        build pytorch sparse tensor from scipy sparse matrix
        reference: https://stackoverflow.com/questions/50665141
        """
        shape = L.shape
        i = torch.LongTensor(np.vstack((L.row, L.col)).astype(int))
        v = torch.FloatTensor(L.data)
        return torch.sparse.FloatTensor(i, v, torch.Size(shape))

    def forward(self, supports, inputs, state, output_size, bias_start=0.0):
        # Reshape input and state to (batch_size, num_nodes,
        # input_dim/hidden_dim)
        batch_size = inputs.shape[0]
        inputs = torch.reshape(inputs, (batch_size, self._num_nodes, -1))
        state = torch.reshape(state, (batch_size, self._num_nodes, -1))
        # (batch, num_nodes, input_dim+hidden_dim)
        inputs_and_state = torch.cat([inputs, state], dim=2)
        input_size = self._input_size

        x0 = inputs_and_state  # (batch, num_nodes, input_dim+hidden_dim)
        # (batch, 1, num_nodes, input_dim+hidden_dim)
        x = torch.unsqueeze(x0, dim=1)

        if self._max_diffusion_step == 0:
            pass
        else:
            for support in supports:
                """x1 = torch.stack([
                    torch.sparse.mm(support, x0[b]) for b in range(batch_size)
                ], dim=0)"""
                x1 = torch.stack([
                    torch.sparse.mm(support, x0[b]) for b in range(batch_size)
                ], dim=0)

                x = self._concat(x, x1)

                xk_minus_two = x0
                xk_minus_one = x1

                for k in range(2, self._max_diffusion_step + 1):
                    """x2 = torch.stack([
                        2 * torch.sparse.mm(support, xk_minus_one[b]) - xk_minus_two[b]
                        for b in range(batch_size)
                    ], dim=0)"""
                    x2 = torch.stack([
                        2 * torch.sparse.mm(support, xk_minus_one[b]) - xk_minus_two[b]
                        for b in range(batch_size)
                    ], dim=0)


                    x = self._concat(x, x2)
                    xk_minus_two, xk_minus_one = xk_minus_one, x2

        num_matrices = len(supports) * \
                       self._max_diffusion_step + 1  # Adds for x itself
        # (batch, num_nodes, num_matrices, input_hidden_size)
        x = torch.transpose(x, dim0=1, dim1=2)
        # (batch, num_nodes, input_hidden_size, num_matrices)
        x = torch.transpose(x, dim0=2, dim1=3)
        x = torch.reshape(
            x,
            shape=[
                batch_size,
                self._num_nodes,
                input_size *
                num_matrices])
        x = torch.reshape(
            x,
            shape=[
                batch_size *
                self._num_nodes,
                input_size *
                num_matrices])
        # (batch_size * self._num_nodes, output_size)
        print(">> DiffusionGraphConv Debug")
        print(f"    inputs.shape = {inputs.shape}")
        print(f"    state.shape  = {state.shape}")
        print(f"    x0.shape     = {x0.shape}")
        print(f"    num_matrices = {num_matrices}")
        print(f"    input_size   = {input_size}")
        print(f"    x.shape before matmul: {x.shape}")
        print(f"    weight.shape: {self.weight.shape}")

        x = torch.matmul(x, self.weight)
        x = torch.add(x, self.biases)
        return torch.reshape(x, [batch_size, self._num_nodes * output_size])


class DCGRUCell(nn.Module):
    """
    Diffusion Convolutional GRU cell.
    """

    def __init__(
            self,
            input_dim,
            num_units,
            max_diffusion_step,
            num_nodes,
            filter_type="laplacian",
            nonlinearity='tanh',
            use_gc_for_ru=True):
        super(DCGRUCell, self).__init__()

        assert use_gc_for_ru, "Only use_gc_for_ru=True is supported. _fc path is not implemented."

        self._activation = torch.tanh if nonlinearity == 'tanh' else torch.relu
        self._num_nodes = num_nodes
        self._num_units = num_units
        self._max_diffusion_step = max_diffusion_step
        self._use_gc_for_ru = use_gc_for_ru

        if filter_type == "laplacian":
            self._num_supports = 1
        elif filter_type == "random_walk":
            self._num_supports = 1
        elif filter_type == "dual_random_walk":
            self._num_supports = 2
        else:
            raise ValueError(f"Unknown filter type: {filter_type}")

        self.dconv_gate = DiffusionGraphConv(
            num_supports=self._num_supports,
            input_dim=input_dim,
            hid_dim=num_units,
            num_nodes=num_nodes,
            max_diffusion_step=max_diffusion_step,
            output_dim=num_units * 2,
            filter_type=filter_type
        )
        self.dconv_candidate = DiffusionGraphConv(
            num_supports=self._num_supports,
            input_dim=input_dim,
            hid_dim=num_units,
            num_nodes=num_nodes,
            max_diffusion_step=max_diffusion_step,
            output_dim=num_units,
            filter_type=filter_type
        )

    @property
    def output_size(self):
        return self._num_nodes * self._num_units

    def forward(self, supports, inputs, state):

        output_size = 2 * self._num_units
        value = torch.sigmoid(self.dconv_gate(supports, inputs, state, output_size, bias_start=1.0))
        value = value.view(-1, self._num_nodes, output_size)
        r, u = torch.split(value, self._num_units, dim=-1)

        r = r.reshape(-1, self._num_nodes * self._num_units)
        u = u.reshape(-1, self._num_nodes * self._num_units)

        c = self.dconv_candidate(supports, inputs, r * state, self._num_units)
        if self._activation is not None:
            c = self._activation(c)

        output = new_state = u * state + (1 - u) * c
        return output, new_state

    def init_hidden(self, batch_size):
        return torch.zeros(batch_size, self._num_nodes * self._num_units)



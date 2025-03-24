import torch
import torch.nn as nn
import torch.nn.functional as F
from DCGRU_cell import DCGRUCell  # your DCGRUCell
import utils  # assume last_relevant_pytorch exists here


class DCGRUClassifier(nn.Module):
    def __init__(self, input_dim, num_nodes, rnn_units, num_layers,
                 max_diffusion_step, num_classes, dcgru_activation='tanh',
                 filter_type='laplacian', dropout=0.3, device=None):
        super(DCGRUClassifier, self).__init__()
        self.num_nodes = num_nodes
        self.rnn_units = rnn_units
        self.num_layers = num_layers
        self.device = device
        self.num_classes = num_classes

        # Build encoder with stacked DCGRU cells
        self.encoder = nn.ModuleList()
        self.encoder.append(
            DCGRUCell(input_dim=input_dim, num_units=rnn_units,
                      max_diffusion_step=max_diffusion_step,
                      num_nodes=num_nodes, nonlinearity=dcgru_activation,
                      filter_type=filter_type)
        )
        for _ in range(1, num_layers):
            self.encoder.append(
                DCGRUCell(input_dim=rnn_units, num_units=rnn_units,
                          max_diffusion_step=max_diffusion_step,
                          num_nodes=num_nodes, nonlinearity=dcgru_activation,
                          filter_type=filter_type)
            )

        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(rnn_units, num_classes)

    def init_hidden(self, batch_size):
        return [torch.zeros(batch_size, self.num_nodes * self.rnn_units).to(self.device)
                for _ in range(self.num_layers)]

    def forward(self, input_seq, seq_lengths, supports):
        """
        input_seq: (batch, seq_len, num_nodes, input_dim)
        seq_lengths: (batch,)
        supports: list of support matrices (sparse or dense)
        """
        batch_size, seq_len = input_seq.shape[0], input_seq.shape[1]

        # (seq_len, batch, num_nodes * input_dim)
        input_seq = input_seq.permute(1, 0, 2, 3).reshape(seq_len, batch_size, -1)

        hidden_states = self.init_hidden(batch_size)

        current_inputs = input_seq
        for i in range(self.num_layers):
            output_inner = []
            state = hidden_states[i]
            for t in range(seq_len):
                _, state = self.encoder[i](supports, current_inputs[t], state)
                output_inner.append(state)
            hidden_states[i] = state
            current_inputs = torch.stack(output_inner, dim=0)

        # (batch_size, seq_len, num_nodes * rnn_units)
        output_seq = current_inputs.permute(1, 0, 2)

        # Extract last relevant hidden state
        last_out = utils.last_relevant_pytorch(output_seq, seq_lengths, batch_first=True)
        last_out = last_out.view(batch_size, self.num_nodes, self.rnn_units)

        # Fully connected prediction layer
        logits = self.fc(self.relu(self.dropout(last_out)))  # (B, N, num_classes)
        pooled_logits, _ = torch.max(logits, dim=1)  # (B, num_classes)

        return pooled_logits

# dcgru_seizure_detection.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import accuracy_score, f1_score
from DataExtract.retrieve import batch_from_files, INCLUDED_CHANNELS
from argparse import Namespace
import numpy as np
import os
import matplotlib.pyplot as plt


class DiffusionGraphConv(nn.Module):
    def __init__(self, num_supports, input_dim, hid_dim, num_nodes, max_diffusion_step, output_dim):
        super(DiffusionGraphConv, self).__init__()
        self._input_size = input_dim + hid_dim
        self._num_nodes = num_nodes
        self._max_diffusion_step = max_diffusion_step
        self._num_supports = num_supports
        self.num_matrices = num_supports * max_diffusion_step + 1

        self.weight = nn.Parameter(torch.FloatTensor(self._input_size * self.num_matrices, output_dim))
        self.biases = nn.Parameter(torch.FloatTensor(output_dim))
        nn.init.xavier_uniform_(self.weight)
        nn.init.constant_(self.biases, 0)

    def forward(self, supports, inputs, state):
        batch_size = inputs.shape[0]
        inputs = inputs.view(batch_size, self._num_nodes, -1)
        state = state.view(batch_size, self._num_nodes, -1)
        x0 = torch.cat([inputs, state], dim=2)
        x = x0.unsqueeze(1)

        for support in supports:
            x1 = torch.stack([torch.sparse.mm(support, x0[b]) for b in range(batch_size)], dim=0)
            x = torch.cat([x, x1.unsqueeze(1)], dim=1)
            xk_minus_two, xk_minus_one = x0, x1

            for _ in range(2, self._max_diffusion_step + 1):
                x2 = torch.stack([
                    2 * torch.sparse.mm(support, xk_minus_one[b]) - xk_minus_two[b]
                    for b in range(batch_size)
                ], dim=0)
                x = torch.cat([x, x2.unsqueeze(1)], dim=1)
                xk_minus_two, xk_minus_one = xk_minus_one, x2

        x = x.permute(0, 2, 3, 1).reshape(batch_size * self._num_nodes, -1)
        x = torch.matmul(x, self.weight) + self.biases
        return x.view(batch_size, self._num_nodes * -1)


class DCGRUCell(nn.Module):
    def __init__(self, input_dim, num_units, max_diffusion_step, num_nodes, num_supports):
        super(DCGRUCell, self).__init__()
        self._num_units = num_units
        self._num_nodes = num_nodes

        self.dconv_gate = DiffusionGraphConv(num_supports, input_dim, num_units, num_nodes, max_diffusion_step, num_units * 2)
        self.dconv_candidate = DiffusionGraphConv(num_supports, input_dim, num_units, num_nodes, max_diffusion_step, num_units)

    def forward(self, supports, inputs, state):
        value = torch.sigmoid(self.dconv_gate(supports, inputs, state))
        value = value.view(-1, self._num_nodes, 2 * self._num_units)
        r, u = torch.split(value, self._num_units, dim=-1)
        r, u = r.reshape(-1, self._num_nodes * self._num_units), u.reshape(-1, self._num_nodes * self._num_units)
        c = self.dconv_candidate(supports, inputs, r * state)
        c = torch.tanh(c)
        output = u * state + (1 - u) * c
        return output, output


class DCGRUClassifier(nn.Module):
    def __init__(self, input_dim, num_nodes, rnn_units, num_layers, max_diffusion_step, num_classes, num_supports):
        super(DCGRUClassifier, self).__init__()
        self.encoder = nn.ModuleList()
        self.encoder.append(DCGRUCell(input_dim, rnn_units, max_diffusion_step, num_nodes, num_supports))
        for _ in range(1, num_layers):
            self.encoder.append(DCGRUCell(rnn_units, rnn_units, max_diffusion_step, num_nodes, num_supports))
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(rnn_units, num_classes)
        self.num_nodes = num_nodes
        self.rnn_units = rnn_units
        self.num_layers = num_layers

    def forward(self, x, supports):
        batch_size, seq_len, num_nodes, input_dim = x.shape
        x = x.permute(1, 0, 2, 3)
        hidden = [torch.zeros(batch_size, self.num_nodes * self.rnn_units).to(x.device) for _ in range(self.num_layers)]

        for t in range(seq_len):
            input_t = x[t].reshape(batch_size, -1)
            for i, cell in enumerate(self.encoder):
                out, hidden[i] = cell(supports, input_t, hidden[i])
                input_t = out

        out = out.view(batch_size, self.num_nodes, self.rnn_units)
        logits = self.fc(self.dropout(out))
        logits, _ = torch.max(logits, dim=1)
        return logits


def load_marker_paths(marker_dir, sz_file, nosz_file):
    def read_paths(file_path):
        with open(file_path, 'r') as f:
            return [(line.strip().split(',')[0], int(line.strip().split(',')[1])) for line in f.readlines() if line.strip().endswith('.h5')]

    seizure_paths = read_paths(sz_file)
    nonseizure_paths = read_paths(nosz_file)
    return seizure_paths + nonseizure_paths

def load_marker_paths(marker_dir, sz_file, nosz_file):
    def read_paths(file_path):
        with open(file_path, 'r') as f:
            return [(line.strip().split(',')[0], int(line.strip().split(',')[1])) for line in f.readlines() if line.strip().endswith('.h5')]

    seizure_paths = read_paths(sz_file)
    nonseizure_paths = read_paths(nosz_file)
    return seizure_paths + nonseizure_paths

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    writer = SummaryWriter(log_dir=os.path.join(args.save_dir or "runs", "DCGRU"))

    model = DCGRUClassifier(
        input_dim=args.input_dim,
        num_nodes=args.num_nodes,
        rnn_units=args.rnn_units,
        num_layers=args.num_rnn_layers,
        max_diffusion_step=args.max_diffusion_step,
        num_classes=args.num_classes,
        num_supports=2 if args.filter_type == 'dual_random_walk' else 1
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr_init, weight_decay=args.l2_wd)
    criterion = nn.BCEWithLogitsLoss()

    marker_dir = os.path.join("DataExtract", "file_markers_detection")
    train_list = load_marker_paths(
        marker_dir,
        os.path.join(marker_dir, "trainSet_seq2seq_12s_sz.txt"),
        os.path.join(marker_dir, "trainSet_seq2seq_12s_nosz.txt")
    )
    print(f">> Loaded {len(train_list)} training samples")

    sensor_ids = [ch.split()[-1] for ch in INCLUDED_CHANNELS]
    label_dict = dict(train_list)

    model.train()
    global_step = 0
    all_losses = []
    all_accs = []
    all_f1s = []

    for epoch in range(args.num_epochs):
        print(f"Epoch {epoch+1}/{args.num_epochs}")
        file_list = []
        for path, _ in train_list:
            full_path = os.path.join(args.input_dir, path)
            if not os.path.exists(full_path):
                print(f"⚠️ Missing file: {full_path}")
            else:
                file_list.append(full_path)
        batch_iter = batch_from_files(
            args.input_dir,
            file_list=file_list,
            batch_size=args.train_batch_size,
            sensor_ids=sensor_ids,
            top_k=args.top_k,
            filter_type=args.filter_type,
            normalization=args.normalization,
            combined=args.combined_graph,
            adj_mat_dir=args.adj_mat_dir,
            shuffle_data=True
        )

        batch_found = False
        for batch in batch_iter:
            batch_found = True
            try:
                batch_x, batch_y_paths, _, batch_supports = batch
            except Exception as e:
                print(f">> Unpacking error: {e}")
                continue

            labels = [label_dict[os.path.basename(p)] for p in batch_y_paths]
            x = torch.FloatTensor(batch_x).to(device)
            y = torch.FloatTensor(labels).unsqueeze(1).to(device)
            supports = batch_supports[0]
            supports = [s.to(device) for s in supports]

            optimizer.zero_grad()
            logits = model(x, supports)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                preds = torch.sigmoid(logits).round()
                acc = accuracy_score(y.cpu().numpy(), preds.cpu().numpy())
                f1 = f1_score(y.cpu().numpy(), preds.cpu().numpy(), zero_division=0)

            print(f"  Step {global_step}: Loss={loss.item():.4f}, Acc={acc:.4f}, F1={f1:.4f}")
            writer.add_scalar("Train/Loss", loss.item(), global_step)
            writer.add_scalar("Train/Accuracy", acc, global_step)
            writer.add_scalar("Train/F1", f1, global_step)
            global_step += 1

            all_losses.append(loss.item())
            all_accs.append(acc)
            all_f1s.append(f1)

        if not batch_found:
            print(f">> No batches yielded in epoch {epoch+1}")

    writer.close()


if __name__ == "__main__":
    from args import get_args
    args = get_args()
    train(args)
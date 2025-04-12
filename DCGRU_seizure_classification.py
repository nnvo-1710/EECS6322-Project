import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import h5py
import argparse
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt

from src.DCGRU_cell import DCGRUCell
from src.utils import seed_torch
from DataExtract.retrieve_classification import batch_generator_from_files


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
        batch_size, seq_len, num_nodes, input_dim = input_seq.shape
        hidden_states = self.init_hidden(batch_size)

        input_seq = input_seq.permute(1, 0, 2, 3)
        current_inputs = input_seq

        for i in range(self.num_layers):
            output_inner = []
            state = hidden_states[i]
            for t in range(seq_len):
                input_t = current_inputs[t]
                input_t = input_t.view(input_t.shape[0], input_t.shape[1] * input_t.shape[2])
                _, state = self.encoder[i](supports, input_t, state)
                output_inner.append(state)
            hidden_states[i] = state
            current_inputs = torch.stack(output_inner, dim=0)

        output_seq = current_inputs.permute(1, 0, 2)
        last_out = output_seq[:, -1, :]
        last_out = last_out.view(batch_size, self.num_nodes, self.rnn_units)

        logits = self.fc(self.relu(self.dropout(last_out)))
        pooled_logits, _ = torch.max(logits, dim=1)

        return pooled_logits


def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    seed_torch(args.seed)


    model = DCGRUClassifier(
        input_dim=100,
        num_nodes=19,
        rnn_units=args.rnn_units,
        num_layers=args.num_layers,
        max_diffusion_step=args.max_diffusion_step,
        num_classes=args.num_classes,
        device=device
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()


    writer = SummaryWriter(log_dir=os.path.join(args.save_dir, "runs"))

    print("Starting training...")
    best_dev_acc = 0
    global_step = 0
    train_losses = []
    val_accuracies = []
    val_f1s = []

    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0
        total = 0


        train_gen = batch_generator_from_files(
            directory=args.train_dir,
            batch_size=args.batch_size,
            shuffle_data=True,
            top_k=5,
            filter_type="laplacian",
            use_combined_graph=False
        )

        for batch_clips, batch_labels, batch_adj_mats, batch_supports in train_gen:
            batch_size_actual = batch_clips.shape[0]
            batch_x = torch.FloatTensor(batch_clips).to(device)
            batch_y = torch.LongTensor(batch_labels).to(device)
            supports = [s.to(device) for s in batch_supports]
            seq_lengths = torch.full((batch_x.shape[0],), batch_x.shape[1], dtype=torch.long).to(device)

            optimizer.zero_grad()
            logits = model(batch_x, seq_lengths, supports)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * batch_size_actual
            total += batch_size_actual

            writer.add_scalar("Train/Loss", loss.item(), global_step)
            global_step += 1

        avg_epoch_loss = epoch_loss / total
        train_losses.append(avg_epoch_loss)
        print(f"Epoch {epoch+1}, Train Loss: {avg_epoch_loss:.4f}")

        # Validation
        model.eval()
        all_preds = []
        all_labels = []

        # Fresh dev generator
        dev_gen = batch_generator_from_files(
            directory=args.dev_dir,
            batch_size=args.batch_size,
            shuffle_data=False,
            top_k=5,
            filter_type="laplacian",
            use_combined_graph=False
        )

        with torch.no_grad():
            for batch_clips, batch_labels, batch_adj_mats, batch_supports in dev_gen:
                batch_x = torch.FloatTensor(batch_clips).to(device)
                batch_y = torch.LongTensor(batch_labels).to(device)
                supports = [s.to(device) for s in batch_supports]
                seq_lengths = torch.full((batch_x.shape[0],), batch_x.shape[1], dtype=torch.long).to(device)

                logits = model(batch_x, seq_lengths, supports)
                preds = torch.argmax(logits, dim=1)

                all_preds.append(preds.cpu().numpy())
                all_labels.append(batch_y.cpu().numpy())

        all_preds = np.concatenate(all_preds)
        all_labels = np.concatenate(all_labels)
        dev_acc = accuracy_score(all_labels, all_preds)
        dev_f1 = f1_score(all_labels, all_preds, average='weighted')

        val_accuracies.append(dev_acc)
        val_f1s.append(dev_f1)

        print(f"Validation Accuracy: {dev_acc:.4f}, F1 Score: {dev_f1:.4f}")

        writer.add_scalar("Val/Accuracy", dev_acc, epoch)
        writer.add_scalar("Val/F1", dev_f1, epoch)

        if dev_acc > best_dev_acc:
            best_dev_acc = dev_acc
            torch.save(model.state_dict(), os.path.join(args.save_dir, "best_model.pth"))

    writer.close()


    plt.figure()
    plt.plot(range(1, len(train_losses) + 1), train_losses, label='Train Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Curve')
    plt.legend()
    plt.savefig(os.path.join(args.save_dir, "train_loss_curve.png"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_dir', type=str, required=True, help='Path to training h5 directory')
    parser.add_argument('--dev_dir', type=str, required=True, help='Path to dev h5 directory')
    parser.add_argument('--save_dir', type=str, required=True, help='Directory to save best model')
    parser.add_argument('--rnn_units', type=int, default=64)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--max_diffusion_step', type=int, default=2)
    parser.add_argument('--num_classes', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    main(args)

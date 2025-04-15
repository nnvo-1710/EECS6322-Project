import numpy as np
import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from utils import seed_torch, get_logger, get_save_dir
from constants import *
from args import get_args
from model import DCGRUClassifier
from DataExtract.retrieve import batch_from_files, INCLUDED_CHANNELS
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
from tensorboardX import SummaryWriter
import glob
from sklearn.model_selection import train_test_split


def main(args):
    args.cuda = torch.cuda.is_available()
    device = torch.device("cuda" if args.cuda else "cpu")
    seed_torch(seed=args.rand_seed)

    args.save_dir = get_save_dir(args.save_dir, training=args.do_train)
    with open(os.path.join(args.save_dir, 'args.json'), 'w') as f:
        json.dump(vars(args), f, indent=4, sort_keys=True)

    log = get_logger(args.save_dir, 'train')
    tbx = SummaryWriter(args.save_dir)
    log.info('Args: {}'.format(json.dumps(vars(args), indent=4, sort_keys=True)))

    # Setup data paths
    train_dir = os.path.join(args.input_dir, 'train', 'clipLen12_timeStepSize1')
    test_dir = os.path.join(args.input_dir, 'eval', 'clipLen12_timeStepSize1')

    sensor_ids = [x.split(' ')[-1] for x in INCLUDED_CHANNELS]

    # Split train files into train/dev
    all_train_files = glob.glob(os.path.join(train_dir, "*.h5"))
    if len(all_train_files) == 0:
        raise ValueError(f"No .h5 files found in {train_dir}. Please check the path or data availability.")
    train_files, dev_files = train_test_split(all_train_files, test_size=0.2, random_state=42)

    # Build model
    model = DCGRUClassifier(
        input_dim=args.input_dim,
        num_nodes=args.num_nodes,
        rnn_units=args.rnn_units,
        num_layers=args.num_rnn_layers,
        max_diffusion_step=args.max_diffusion_step,
        num_classes=args.num_classes,
        dcgru_activation=args.dcgru_activation,
        filter_type=args.filter_type,
        dropout=args.dropout,
        device=device
    ).to(device)

    if args.do_train:
        optimizer = optim.Adam(model.parameters(), lr=args.lr_init, weight_decay=args.l2_wd)
        scheduler = CosineAnnealingLR(optimizer, T_max=args.num_epochs)
        loss_fn = nn.CrossEntropyLoss()
        best_val_loss = float('inf')
        patience_counter = 0

        for epoch in range(1, args.num_epochs + 1):
            model.train()
            total_loss = 0
            train_gen = batch_from_files(
                directory=train_dir,
                file_list=train_files,
                batch_size=args.train_batch_size,
                sensor_ids=sensor_ids,
                top_k=args.top_k,
                combined=args.combined_graph,
                adj_mat_dir=args.adj_mat_dir,
                filter_type=args.filter_type,
                normalization=args.normalization
            )
            with tqdm(train_gen, desc=f"Epoch {epoch}") as tbar:
                for x, y, _, supports in tbar:
                    x = torch.tensor(x).float().to(device)
                    y = torch.tensor(y).long().to(device).view(-1)
                    seq_lengths = torch.tensor([x.shape[1]] * x.shape[0]).to(device)
                    supports = [s.to(device) for s_list in supports for s in s_list]

                    optimizer.zero_grad()
                    print(f" Input x shape: {x.shape}, seq_lengths: {seq_lengths.tolist()}")
                    print(f">> Number of supports: {len(supports)}")
                    logits = model(x, seq_lengths, supports[0])
                    loss = loss_fn(logits, y)
                    loss.backward()
                    optimizer.step()

                    total_loss += loss.item()
                    tbar.set_postfix(loss=loss.item())

                    scheduler.step()
                    avg_loss = total_loss / (tbar.n or 1)
                    tbx.add_scalar('train/loss', avg_loss, epoch)

                    # Validation
                    val_loss = evaluate(model, dev_files, sensor_ids, args, device)
                    tbx.add_scalar('val/loss', val_loss, epoch)

                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        patience_counter = 0
                        torch.save(model.state_dict(), os.path.join(args.save_dir, 'best_model.pt'))
                    else:
                        patience_counter += 1
                        if patience_counter >= args.patience:
                            log.info("Early stopping triggered.")
                            break

    # Final Evaluation
    model.load_state_dict(torch.load(os.path.join(args.save_dir, 'best_model.pt')))
    evaluate(model, test_dir, sensor_ids, args, device, is_test=True)


def evaluate(model, data_source, sensor_ids, args, device, is_test=False):
    model.eval()
    all_preds, all_labels = [], []
    loss_fn = nn.CrossEntropyLoss()
    total_loss = 0

    if isinstance(data_source, list):
        data_gen = batch_from_files(
            directory=args.input_dir,
            file_list=data_source,
            batch_size=args.test_batch_size,
            sensor_ids=sensor_ids,
            top_k=args.top_k,
            combined=args.combined_graph,
            adj_mat_dir=args.adj_mat_dir,
            filter_type=args.filter_type,
            normalization=args.normalization,
            shuffle_data=False
        )
    else:
        data_gen = batch_from_files(
            directory=data_source,
            batch_size=args.test_batch_size,
            sensor_ids=sensor_ids,
            top_k=args.top_k,
            combined=args.combined_graph,
            adj_mat_dir=args.adj_mat_dir,
            filter_type=args.filter_type,
            normalization=args.normalization,
            shuffle_data=False
        )

    with torch.no_grad():
        for x, y, _, supports in data_gen:
            x = torch.tensor(x).float().to(device)
            y = torch.tensor(y).long().to(device).view(-1)
            seq_lengths = torch.tensor([x.shape[1]] * x.shape[0]).to(device)
            supports = [[s.to(device) for s in s_list] for s_list in supports]

            logits = model(x, seq_lengths, supports)
            loss = loss_fn(logits, y)
            total_loss += loss.item()

            preds = torch.argmax(F.softmax(logits, dim=1), dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())

    acc = np.mean(np.array(all_preds) == np.array(all_labels))
    avg_loss = total_loss / (len(all_preds) // args.test_batch_size + 1)
    print(f"Eval ({'Test' if is_test else 'Dev'}) - Loss: {avg_loss:.4f}, Acc: {acc:.4f}")
    return avg_loss


if __name__ == '__main__':
    args = get_args()
    main(args)

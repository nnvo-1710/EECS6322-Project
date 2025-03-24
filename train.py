import numpy as np
import os
import pickle
import torch
import json
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import utils
from data.data_utils import *
from data.dataloader_classification import load_dataset_classification
from constants import *
from args import get_args
from collections import OrderedDict
from model import DCGRUClassifier
from tensorboardX import SummaryWriter
from tqdm import tqdm
from torch.optim.lr_scheduler import CosineAnnealingLR
import copy

def main(args):
    args.cuda = torch.cuda.is_available()
    device = "cuda" if args.cuda else "cpu"
    utils.seed_torch(seed=args.rand_seed)

    args.save_dir = utils.get_save_dir(args.save_dir, training=args.do_train)
    args_file = os.path.join(args.save_dir, 'args.json')
    with open(args_file, 'w') as f:
        json.dump(vars(args), f, indent=4, sort_keys=True)

    log = utils.get_logger(args.save_dir, 'train')
    tbx = SummaryWriter(args.save_dir)
    log.info('Args: {}'.format(json.dumps(vars(args), indent=4, sort_keys=True)))

    # Load dataset
    dataloaders, _, scaler = load_dataset_classification(
        input_dir=args.input_dir,
        raw_data_dir=args.raw_data_dir,
        train_batch_size=args.train_batch_size,
        test_batch_size=args.test_batch_size,
        time_step_size=args.time_step_size,
        max_seq_len=args.max_seq_len,
        standardize=True,
        num_workers=args.num_workers,
        padding_val=0.,
        augmentation=args.data_augment,
        adj_mat_dir='./data/electrode_graph/adj_mx_3d.pkl',
        graph_type=args.graph_type,
        top_k=args.top_k,
        filter_type=args.filter_type,
        use_fft=args.use_fft,
        preproc_dir=args.preproc_dir)

    # Build model
    log.info('Building model...')
    if args.model_name == "dcgru_classifier":
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
        )
    else:
        raise NotImplementedError(f"Model {args.model_name} is not supported.")

    model = model.to(device)

    if args.do_train:
        optimizer = optim.Adam(model.parameters(), lr=args.lr_init, weight_decay=args.l2_wd)
        scheduler = CosineAnnealingLR(optimizer, T_max=args.num_epochs)
        loss_fn = nn.CrossEntropyLoss()

        best_val_loss = float('inf')
        patience_counter = 0

        for epoch in range(1, args.num_epochs + 1):
            model.train()
            total_loss = 0
            with tqdm(dataloaders['train'], desc=f"Epoch {epoch}") as tbar:
                for x, y, seq_lengths, supports, _, _ in tbar:
                    x, y = x.to(device), y.to(device).view(-1)
                    seq_lengths = seq_lengths.to(device)
                    supports = [s.to(device) for s in supports]

                    optimizer.zero_grad()
                    logits = model(x, seq_lengths, supports)
                    loss = loss_fn(logits, y)
                    loss.backward()
                    optimizer.step()

                    total_loss += loss.item()
                    tbar.set_postfix(loss=loss.item())

            scheduler.step()
            avg_loss = total_loss / len(dataloaders['train'])
            tbx.add_scalar('train/loss', avg_loss, epoch)

            # Validation
            val_loss = evaluate(model, dataloaders['dev'], args, device)
            tbx.add_scalar('val/loss', val_loss, epoch)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                utils.save_model(model, os.path.join(args.save_dir, 'best_model.pt'))
            else:
                patience_counter += 1
                if patience_counter >= args.patience:
                    log.info("Early stopping triggered.")
                    break

    # Load best model for test evaluation
    model.load_state_dict(torch.load(os.path.join(args.save_dir, 'best_model.pt')))
    evaluate(model, dataloaders['test'], args, device, is_test=True)

def evaluate(model, dataloader, args, device, is_test=False):
    model.eval()
    all_preds, all_labels = [], []
    loss_fn = nn.CrossEntropyLoss()
    total_loss = 0

    with torch.no_grad():
        for x, y, seq_lengths, supports, _, _ in dataloader:
            x, y = x.to(device), y.to(device).view(-1)
            seq_lengths = seq_lengths.to(device)
            supports = [s.to(device) for s in supports]

            logits = model(x, seq_lengths, supports)
            loss = loss_fn(logits, y)
            total_loss += loss.item()

            preds = torch.argmax(F.softmax(logits, dim=1), dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())

    acc = np.mean(np.array(all_preds) == np.array(all_labels))
    avg_loss = total_loss / len(dataloader)
    print(f"Eval ({'Test' if is_test else 'Dev'}) - Loss: {avg_loss:.4f}, Acc: {acc:.4f}")
    return avg_loss


if __name__ == '__main__':
    args = get_args()
    main(args)

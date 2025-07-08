import argparse
import os
import time
import torch
import numpy as np
import random
from dgl.data import register_data_args
from utils.load_dataset import load_data
from utils.early import EarlyStopping
from utils.certainty import uncertainty
from model.myMamba_model import myMamba, ModelArgs
from tqdm import trange


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_graph_and_features(args):
    g, features, labels, train_mask, val_mask, test_mask, in_feats, n_classes, n_edges = load_data(
        name=args.dataset, seed=args.seed, self_loop=args.self_loop)
    if args.gpu >= 0:
        features = features.cuda()
        labels = labels.cuda()
        train_mask = train_mask.cuda()
        val_mask = val_mask.cuda()
        test_mask = test_mask.cuda()
        g = g.to(args.gpu)
    return g, features, labels, train_mask, val_mask, test_mask, in_feats, n_classes, n_edges


def load_or_compute_spectral(g, args):
    a_path = f'./mid_data/{args.dataset}/a_.pkl'
    u_path = f'./mid_data/{args.dataset}/U.pkl'

    try:
        a_ = torch.load(a_path).to(args.gpu)
        U = torch.load(u_path).cpu()
    except:
        from networkx import normalized_laplacian_matrix
        g1 = g.cpu().to_networkx().to_undirected()
        A = g.adj().to_dense()
        D_ = np.diag(np.where(np.isinf(A.sum(1).cpu().numpy() ** -0.5), 0, A.sum(1).cpu().numpy() ** -0.5))
        A_norm = torch.mm(torch.from_numpy(D_), torch.mm(A.cpu(), torch.from_numpy(D_))).cuda()

        L = normalized_laplacian_matrix(g1).todense()
        a_, b_ = np.linalg.eigh(L)
        a_ = torch.from_numpy(a_.astype(np.float32)).to(args.gpu)
        U = torch.from_numpy(b_.astype(np.float32))
        return A_norm, a_, U

    A = g.adj().to_dense()
    D_ = np.diag(np.where(np.isinf(A.sum(1).cpu().numpy() ** -0.5), 0, A.sum(1).cpu().numpy() ** -0.5))
    A_norm = torch.mm(torch.from_numpy(D_), torch.mm(A.cpu(), torch.from_numpy(D_))).cuda()
    return A_norm, a_, U


def evaluate(model, features, U, a_, labels, mask):
    model.eval()
    with torch.no_grad():
        logits = model(features, U, a_)[mask]
        labels = labels[mask]
        _, indices = torch.max(logits, dim=1)
        correct = torch.sum(indices == labels)
        return correct.item() / len(labels)


def train(args):
    seed_everything(args.seed)
    g, features, labels, train_mask, val_mask, test_mask, in_feats, n_classes, n_edges = load_graph_and_features(args)
    A, a_, U = load_or_compute_spectral(g, args)

    model_args = ModelArgs(args.dataset, g.num_nodes(), args.d_model, args.n_hidden, args.n_layers,
                           args.ssm_layers, in_feats, n_classes, args.dropout, args.weight)
    model = myMamba(model_args).cuda() if args.gpu >= 0 else myMamba(model_args)
    A, U = A.cuda(), U.cuda()

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = torch.nn.CrossEntropyLoss()
    stopper = EarlyStopping(patience=1000)

    best_val, best_test = 0, 0
    dur = []

    for epoch in trange(args.epochs, desc="Training"):
        model.train()
        if epoch >= 3:
            t0 = time.time()

        pred = model(features, U, a_)
        loss = criterion(pred[train_mask], labels[train_mask])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch >= 3:
            dur.append(time.time() - t0)

        train_acc = evaluate(model, features, U, a_, labels, train_mask)
        val_acc = evaluate(model, features, U, a_, labels, val_mask)

        if val_acc > best_val:
            best_val = val_acc
            best_test = evaluate(model, features, U, a_, labels, test_mask)

        print(f"Epoch {epoch:03d} | Loss: {loss.item():.4f} | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f} | "
              f"Speed: {n_edges / np.mean(dur) / 1000 if dur else 0:.2f} KTEPS")

        if stopper.step(loss.item()):
            break

    print(f"Final Test Accuracy: {best_test:.4f}")
    return best_test


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Optimized SSM")
    register_data_args(parser)
    parser.add_argument("--dataset", type=str, default='cora')
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-1)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--n-hidden", type=int, default=16)
    parser.add_argument("--d-model", type=int, default=8)
    parser.add_argument("--weight", type=float, default=1.)
    parser.add_argument("--n-layers", type=int, default=1)
    parser.add_argument("--ssm-layers", type=int, default=1)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--seed", type=int, default=3280387012)
    parser.add_argument("--self-loop", action="store_true")
    parser.set_defaults(self_loop=False)
    args = parser.parse_args()

    seed_everything(args.seed)
    acc = train(args)

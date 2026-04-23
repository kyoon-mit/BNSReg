"""
Signal vs Background classifier using predicted gravitational-wave parameters.
Features: M_chirp_pred, M_ratio_pred, sigma_M_chirp_pred, sigma_M_ratio_pred
"""

import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import wandb

# ── Args ──────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument('--signal',        default='signal.csv')
parser.add_argument('--background',    default='background.csv')
parser.add_argument('--epochs',        type=int,   default=50)
parser.add_argument('--batch_size',    type=int,   default=256)
parser.add_argument('--lr',            type=float, default=5e-4)
parser.add_argument('--eta_min',       type=float, default=1e-6)
parser.add_argument('--t_0',           type=int,   default=40)
parser.add_argument('--t_mult',        type=int,   default=1)
parser.add_argument('--layers',        type=int,   default=2)
parser.add_argument('--hidden',        type=int,   default=64)
parser.add_argument('--test_frac',     type=float, default=0.2)
parser.add_argument('--patience',      type=int,   default=60)
parser.add_argument('--seed',          type=int,   default=42)
parser.add_argument('--wandb_project', default='gw-classifier')
args = parser.parse_args()

torch.manual_seed(args.seed)
rng = np.random.default_rng(args.seed)

FEATURES = ['M_chirp_pred', 'M_ratio_pred', 'sigma_M_chirp_pred', 'sigma_M_ratio_pred']

# ── Data ──────────────────────────────────────────────────────────────────────
sig = pd.read_csv(args.signal)[FEATURES].assign(label=1.0)
bkg = pd.read_csv(args.background)[FEATURES].assign(label=0.0)
df  = pd.concat([sig, bkg], ignore_index=True).dropna()

X = df[FEATURES].values.astype(np.float32)
y = df['label'].values.astype(np.float32)

# Stratified split (manual)
def stratified_split(X, y, test_frac, rng):
    idx0 = np.where(y == 0)[0]; idx1 = np.where(y == 1)[0]
    rng.shuffle(idx0);          rng.shuffle(idx1)
    n0_test = max(1, int(len(idx0) * test_frac))
    n1_test = max(1, int(len(idx1) * test_frac))
    test_idx  = np.concatenate([idx0[:n0_test],  idx1[:n1_test]])
    train_idx = np.concatenate([idx0[n0_test:],  idx1[n1_test:]])
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]

X_train, X_test, y_train, y_test = stratified_split(X, y, args.test_frac, rng)

# StandardScaler (manual)
mu    = X_train.mean(axis=0)
sigma = X_train.std(axis=0) + 1e-8
X_train = (X_train - mu) / sigma
X_test  = (X_test  - mu) / sigma

def to_loader(X, y, shuffle=True):
    ds = TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
    return DataLoader(ds, batch_size=args.batch_size, shuffle=shuffle)

train_loader = to_loader(X_train, y_train, shuffle=True)
test_loader  = to_loader(X_test,  y_test,  shuffle=False)

# ── Model ─────────────────────────────────────────────────────────────────────
class Classifier(nn.Module):
    def __init__(self, in_dim, hidden, n_layers):
        super().__init__()
        blocks = [nn.Linear(in_dim, hidden), nn.CELU(), nn.BatchNorm1d(hidden)]
        for _ in range(n_layers - 1):
            blocks += [nn.Linear(hidden, hidden), nn.CELU(), nn.BatchNorm1d(hidden)]
        blocks.append(nn.Linear(hidden, 1))
        self.net = nn.Sequential(*blocks)

    def forward(self, x):
        return self.net(x).squeeze(1)

device    = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model     = Classifier(len(FEATURES), args.hidden, args.layers).to(device)
opt       = torch.optim.Adam(model.parameters(), lr=args.lr)
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
    opt, T_0=args.t_0, T_mult=args.t_mult, eta_min=args.eta_min
)
loss_fn   = nn.BCEWithLogitsLoss()

# ── W&B ───────────────────────────────────────────────────────────────────────
wandb.init(
    project=args.wandb_project,
    config=vars(args),
    name=f'h{args.hidden}_l{args.layers}_lr{args.lr}_bs{args.batch_size}',
)
wandb.watch(model, log='gradients', log_freq=50)

# ── Training loop ─────────────────────────────────────────────────────────────
def run_epoch(loader, train=True):
    model.train(train)
    total_loss, correct, n = 0.0, 0, 0
    with torch.set_grad_enabled(train):
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            loss   = loss_fn(logits, yb)
            if train:
                opt.zero_grad(); loss.backward(); opt.step()
            total_loss += loss.item() * len(yb)
            correct    += ((logits > 0) == yb.bool()).sum().item()
            n          += len(yb)
    return total_loss / n, correct / n

best_val_loss    = float('inf')
patience_counter = 0

for epoch in range(1, args.epochs + 1):
    tr_loss, tr_acc = run_epoch(train_loader, train=True)
    va_loss, va_acc = run_epoch(test_loader,  train=False)
    scheduler.step()
    wandb.log({'epoch':      epoch,
               'train/loss': tr_loss, 'train/acc': tr_acc,
               'val/loss':   va_loss, 'val/acc':   va_acc,
               'lr':         scheduler.get_last_lr()[0]})
    if epoch % 10 == 0:
        print(f'[{epoch:3d}/{args.epochs}]  '
              f'train loss={tr_loss:.4f} acc={tr_acc:.3f}  |  '
              f'val   loss={va_loss:.4f} acc={va_acc:.3f}  |  '
              f'lr={scheduler.get_last_lr()[0]:.2e}')

    if va_loss < best_val_loss:
        best_val_loss    = va_loss
        patience_counter = 0
        torch.save(model.state_dict(), 'best_model.pt')
    else:
        patience_counter += 1
        if patience_counter >= args.patience:
            print(f'\nEarly stopping at epoch {epoch} (patience={args.patience})')
            break

# reload best weights before eval
model.load_state_dict(torch.load('best_model.pt', weights_only=True))

# ── ROC curve (pure NumPy) ────────────────────────────────────────────────────
model.eval()
all_scores, all_labels = [], []
with torch.no_grad():
    for xb, yb in test_loader:
        scores = torch.sigmoid(model(xb.to(device))).cpu().numpy()
        all_scores.append(scores)
        all_labels.append(yb.numpy())

scores = np.concatenate(all_scores)
labels = np.concatenate(all_labels)

def roc_auc_numpy(labels, scores, n_thresholds=20000):
    thresholds = np.linspace(1, 0, n_thresholds)
    fprs, tprs = [], []
    for t in thresholds:
        tp = np.sum((scores > t) & (labels == 1))
        fp = np.sum((scores > t) & (labels == 0))
        fn = np.sum((scores <= t) & (labels == 1))
        tn = np.sum((scores <= t) & (labels == 0))
        tprs.append(tp / (tp + fn) if (tp + fn) > 0 else 0.0)
        fprs.append(fp / (fp + tn) if (fp + tn) > 0 else 0.0)
    fprs, tprs = np.array(fprs), np.array(tprs)
    roc_auc = float(np.trapezoid(tprs, fprs))
    return fprs, tprs, roc_auc

fpr, tpr, roc_auc = roc_auc_numpy(labels, scores)

fig, ax = plt.subplots(figsize=(8, 6))
ax.semilogx(fpr, tpr, color='darkorange', lw=2, label=f'AUC = {roc_auc:.4f}')
ax.set_xlabel('False Positive Rate (log scale)')
ax.set_ylabel('True Positive Rate')
ax.legend(loc='lower right')
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig('roc_curve.png', dpi=300)
wandb.log({'roc_auc': roc_auc, 'roc_curve': wandb.Image(fig)})

# ── Score distribution ────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 6))
bins = np.linspace(0, 1, 50)
ax.hist(scores[labels == 1], bins=bins, histtype='step', color='red',  lw=2, label='Signal',     density=True)
ax.hist(scores[labels == 0], bins=bins, histtype='step', color='blue', lw=2, label='Background', density=True)
ax.set_xlabel('NN Score')
ax.set_ylabel('Density')
ax.legend()
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig('score_distribution.png', dpi=300)
wandb.log({'score_distribution': wandb.Image(fig)})

print(f'\nROC AUC = {roc_auc:.4f}  →  roc_curve.png saved')

wandb.finish()
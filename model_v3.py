import os
import random
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from process_data import build_index, train_valid_split, PairDataset  # uses your existing code
from utils import ndcg_at_k, write_results_to_file                   # reuse your NDCG + writer


@dataclass
class BPRArgs:
    valid_ratio: float = 0.1
    seed: int = 42

    # training
    num_neg: int = 4          # how many negatives per positive in PairDataset
    batch_size: int = 2048
    epochs: int = 40
    lr: float = 5e-4
    emb_dim: int = 64
    l2: float = 1e-5          # L2 regularization

    topk: int = 20
    pop_neg: bool = True      # use popularity-weighted negatives

    cpu: bool = False
    outdir: str = "./checkpoints_bpr"
    max_eval_users: int = 10000


class BPRMF(nn.Module):
    """
    Simple BPR-MF model: s(u, i) = <p_u, q_i>
    """
    def __init__(self, num_users, num_items, emb_dim=64):
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items

        self.user_emb = nn.Embedding(num_users, emb_dim)
        self.item_emb = nn.Embedding(num_items, emb_dim)

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.user_emb.weight, std=0.01)
        nn.init.normal_(self.item_emb.weight, std=0.01)

    def forward(self, users, items):
        """
        users, items: Long tensors of same shape
        returns: scores [B]
        """
        u = self.user_emb(users)
        i = self.item_emb(items)
        return (u * i).sum(dim=-1)

    @torch.no_grad()
    def predict_users(self, user_ids, chunk=4096):
        """
        Compute scores for many users against all items.
        user_ids: LongTensor [U]
        returns: FloatTensor [U, num_items]
        """
        self.eval()
        device = self.user_emb.weight.device
        user_ids = user_ids.to(device)

        U = user_ids.shape[0]
        num_items = self.num_items
        scores_list = []

        # precompute user embeddings
        u_emb = self.user_emb(user_ids)  # [U, D]

        # items in chunks to save memory
        all_item_emb = self.item_emb.weight  # [I, D]

        for start in range(0, num_items, chunk):
            end = min(start + chunk, num_items)
            item_chunk = all_item_emb[start:end]  # [C, D]

            # scores: [U, C] = u_emb @ item_chunk^T
            sc = torch.matmul(u_emb, item_chunk.t())
            scores_list.append(sc)

        return torch.cat(scores_list, dim=1)  # [U, num_items]


def get_item_pop_probs(train_ui, num_items, eps=1e-12):
    """
    Compute popularity distribution over items for negative sampling.
    """
    cnt = np.zeros(num_items, dtype=np.float64)
    for items in train_ui.values():
        for i in items:
            cnt[i] += 1
    if cnt.sum() == 0:
        return None
    p = cnt / cnt.sum()
    p = (p + eps) / (p + eps).sum()
    return p


def train_epoch_bpr(model, loader, optimizer, device, l2=1e-5):
    model.train()
    total_loss = 0.0
    for users, pos, negs in tqdm(loader, desc="Training Progress"):
        users = users.long().to(device)
        pos = pos.long().to(device)
        negs = negs.long().to(device)

        # pick one negative per positive (first column)
        neg = negs[:, 0]

        # scores
        s_pos = model(users, pos)
        s_neg = model(users, neg)

        # BPR loss: -log sigma(s_pos - s_neg)
        diff = s_pos - s_neg
        loss_bpr = -torch.log(torch.sigmoid(diff) + 1e-12).mean()

        # L2 regularization on embeddings for users, pos, neg
        reg = (
            model.user_emb(users).pow(2).sum(dim=1)
            + model.item_emb(pos).pow(2).sum(dim=1)
            + model.item_emb(neg).pow(2).sum(dim=1)
        ).mean()

        loss = loss_bpr + l2 * reg

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / max(1, len(loader))


@torch.no_grad()
def evaluate_bpr(model, train_ui, valid_ui, K=20, batch_users=1024, device="cpu", max_eval_users=10000):
    """
    Evaluate NDCG@K on valid_ui (user -> set of items).
    """
    model.eval()
    users = [u for u, items in valid_ui.items() if len(items) > 0]
    if not users:
        return 0.0

    if len(users) > max_eval_users:
        users = random.sample(users, max_eval_users)

    ndcgs = []
    num_items = model.num_items

    for s in tqdm(range(0, len(users), batch_users), desc="Evaluate Progress"):
        batch = users[s:s + batch_users]
        u_t = torch.tensor(batch, dtype=torch.long, device=device)

        scores = model.predict_users(u_t)  # [B, I]
        scores = scores.cpu().numpy()

        for r, u in enumerate(batch):
            seen = train_ui.get(u, set())
            if seen:
                scores[r, list(seen)] = -1e9

            # top-K index
            topk = np.argpartition(-scores[r], K)[:K]
            topk = topk[np.argsort(-scores[r][topk])]

            ndcgs.append(ndcg_at_k(topk.tolist(), valid_ui[u], K))

    return float(np.mean(ndcgs))


@torch.no_grad()
def generate_submission_fast(
    model,
    seen_ui,
    num_users,
    num_items,
    idx2user,
    idx2item,
    device,
    K=20,
    user_batch_size=1024,
    out_path="output.txt",
):
    """
    Same idea as in model_v2_neuralmf.py: generate top-K recommendations
    for all users and write to file.
    """
    model.eval()
    all_users = list(range(num_users))
    results = []

    for s in tqdm(range(0, num_users, user_batch_size), desc="Predict Progress"):
        batch_users = all_users[s:s + user_batch_size]
        u_t = torch.tensor(batch_users, dtype=torch.long, device=device)

        scores = model.predict_users(u_t)        # [B, I]
        scores = scores.cpu().numpy()

        for row_idx, u in enumerate(batch_users):
            seen = seen_ui.get(u, set())
            if seen:
                scores[row_idx, list(seen)] = -1e9

            topk = np.argpartition(-scores[row_idx], K)[:K]
            topk = topk[np.argsort(-scores[row_idx][topk])]

            user_orig = idx2user[u]
            rec_items = [idx2item[i] for i in topk]

            results.append([user_orig] + rec_items)

    write_results_to_file(results, out_path)
    print(f"Saved submission to {out_path}")


if __name__ == "__main__":
    args = BPRArgs()

    # 1) Build index and splits (reuse your pipeline)
    ui_pos, user2idx, idx2user, item2idx, idx2item = build_index()
    train_ui, valid_ui = train_valid_split(ui_pos, valid_ratio=args.valid_ratio, seed=args.seed)

    num_users = len(user2idx)
    num_items = len(item2idx)

    # 2) Popularity-weighted negatives (optional but usually helpful)
    pop_p = get_item_pop_probs(train_ui, num_items) if args.pop_neg else None

    dataset = PairDataset(train_ui, num_items, num_neg=args.num_neg, pop_weights=pop_p)

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    num_workers = 4 if device.type == "cuda" else 0
    pin_memory = device.type == "cuda"

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    model = BPRMF(num_users, num_items, emb_dim=args.emb_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_ndcg, best_path = 0.0, None
    os.makedirs(args.outdir, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}")
        loss = train_epoch_bpr(model, loader, optimizer, device, l2=args.l2)
        n20 = evaluate_bpr(model, train_ui, valid_ui, K=args.topk, device=device, max_eval_users=args.max_eval_users)
        print(f"Epoch {epoch:02d} | loss={loss:.4f} | NDCG@{args.topk}={n20:.4f}")

        if n20 > best_ndcg:
            best_ndcg = n20
            best_path = os.path.join(args.outdir, "bprmf_best.pt")
            torch.save(
                {
                    "model": model.state_dict(),
                    "user2idx": user2idx,
                    "item2idx": item2idx,
                    "idx2user": idx2user,
                    "idx2item": idx2item,
                },
                best_path,
            )

    # Load best model and generate final submission on train+valid seen filter
    if best_path:
        ckpt = torch.load(best_path, map_location=device)
        model.load_state_dict(ckpt["model"])

    seen_all = {u: set(train_ui.get(u, set())) | set(valid_ui.get(u, set())) for u in range(num_users)}

    generate_submission_fast(
        model=model,
        seen_ui=seen_all,
        num_users=num_users,
        num_items=num_items,
        idx2user=idx2user,
        idx2item=idx2item,
        device=device,
        K=args.topk,
        user_batch_size=1024,
        out_path="output.txt",
    )
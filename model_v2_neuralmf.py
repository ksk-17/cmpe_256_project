import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from utils import ndcg_at_k, write_results_to_file
from process_data import build_index, train_valid_split, PairDataset
from dataclasses import dataclass
from tqdm import tqdm
import os
import random

@dataclass
class NeuralMFArgs:
    valid_ratio = 0.1
    seed = 42
    num_neg = 4
    batch_size = 1024
    epochs = 16
    lr = 1e-4

    emb_mf = 16
    emb_mlp = 16

    topk = 20
    pop_neg = False

    # Added for correct device/output handling
    cpu = False
    outdir = "./checkpoints"


class NeuralMF(nn.Module):
    def __init__(self, num_users, num_items, f=32, mlp_dim=32):
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items

        # GMF layers
        self.user_mf = nn.Embedding(num_users, f)
        self.item_mf = nn.Embedding(num_items, f)

        # MLP layers
        self.user_mlp = nn.Embedding(num_users, mlp_dim)
        self.item_mlp = nn.Embedding(num_items, mlp_dim)

        self.mlp = nn.Sequential(
            nn.Linear(2 * mlp_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU()
        )
        # Prediction layer (GMF_dim + MLP_output_dim)
        self.predict = nn.Linear(f + 16, 1)
        self.sigmoid = nn.Sigmoid()
        self.init_weights()  # <-- FIX 2: Corrected weight initialization call

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.01)
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, users, items):
        # GMF Path: Element-wise product
        mf = self.user_mf(users) * self.item_mf(items)

        # MLP Path: Concatenate embeddings and pass through MLP
        mlp_input = torch.cat([self.user_mlp(users), self.item_mlp(items)], dim=-1)  # <-- FIX 3: Corrected MLP input
        mlp = self.mlp(mlp_input)

        # Combine GMF and MLP, then predict
        x = torch.cat([mf, mlp], dim=-1)
        logits = self.predict(x).squeeze(-1)
        return logits

    def predict_users(self, user_ids, chunk=4096):
        u = user_ids.shape[0]
        num_items = self.num_items
        scores = []
        device = self.user_mf.weight.device  # Determine model device

        # compute user mf and mlp
        u_mf = self.user_mf(user_ids)
        u_mlp = self.user_mlp(user_ids)

        # Get item embeddings from model's device
        all_item_mf = self.item_mf.weight
        all_item_mlp = self.item_mlp.weight

        for start in range(0, num_items, chunk):
            end = min(start + chunk, num_items)

            # Use pre-computed item weights on the correct device
            item_mf_chunk = all_item_mf[start:end]
            item_mlp_chunk = all_item_mlp[start:end]

            # GMF part
            mf = u_mf.unsqueeze(1) * item_mf_chunk.unsqueeze(0)

            # MLP part
            mlp_in = torch.cat([
                u_mlp.unsqueeze(1).expand(-1, end - start, -1),
                item_mlp_chunk.unsqueeze(0).expand(u, -1, -1)
            ], dim=-1)

            mlp = self.mlp(mlp_in)

            x = torch.cat([mf, mlp], dim=-1)
            logits = self.predict(x).squeeze(-1)
            scores.append(logits)

        return torch.cat(scores, dim=-1)


def get_item_pop_probs(train_ui, num_items, eps=1e-12):
    cnt = np.zeros(num_items, dtype=np.float64)
    for items in train_ui.values():
        for i in items: cnt[i] += 1
    if cnt.sum() == 0:
        return None
    p = cnt / cnt.sum()
    p = (p + eps) / (p + eps).sum()
    return p


def train_epoch(model, loader, optimizer, device, bce_loss):  # <-- FIX 5: Added device
    model.train()
    total_loss = 0.0
    for users, pos, negs in tqdm(loader, "Training Progress"):
        users = users.long().to(device)  # <-- FIX 5: Transfer to device
        pos = pos.long().to(device)  # <-- FIX 5: Transfer to device
        negs = negs.long().to(device)  # <-- FIX 5: Transfer to device
        B, K = negs.shape

        # Positive pairs
        logits_pos = model(users, pos)

        # Negative pairs
        users_neg = users.unsqueeze(1).expand(-1, K).reshape(-1)
        negs_flat = negs.reshape(-1)
        logits_neg = model(users_neg, negs_flat).view(B, K)

        y_pos = torch.ones_like(logits_pos)
        y_neg = torch.zeros_like(logits_neg)

        # Loss calculation (BPR-style loss where y_pos=1, y_neg=0)
        loss = bce_loss(logits_pos, y_pos) + bce_loss(logits_neg, y_neg.float())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / max(1, len(loader))


@torch.no_grad()
def evaluate(model, train_ui, valid_ui, K=20, batch_users=1024, device="cpu", max_eval_users=10000):
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
        scores = model.predict_users(u_t)
        scores = scores.cpu().numpy()

        for r, u in enumerate(batch):
            # filter training items (don’t recommend already seen)
            seen = train_ui.get(u, set())
            if seen:
                scores[r, list(seen)] = -1e9

            # Get top K indices
            topk = np.argpartition(-scores[r], K)[:K]
            # Re-sort the top K for correct order
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
    out_path="submission.txt",
):
    """
    Generate top-K recommendations for all users using the best model
    and write them to a file in a fast, batched way.
    """
    model.eval()
    all_users = list(range(num_users))
    results = []

    for s in tqdm(range(0, num_users, user_batch_size), desc="Predict Progress"):
        batch_users = all_users[s:s + user_batch_size]
        u_t = torch.tensor(batch_users, dtype=torch.long, device=device)

        # scores: (batch_size, num_items)
        scores = model.predict_users(u_t)
        scores = scores.cpu().numpy()

        for row_idx, u in enumerate(batch_users):
            # mask all seen items (train + valid)
            seen = seen_ui.get(u, set())
            if seen:
                scores[row_idx, list(seen)] = -1e9

            # get top-K item indices
            topk = np.argpartition(-scores[row_idx], K)[:K]
            topk = topk[np.argsort(-scores[row_idx][topk])]

            # map back to original IDs
            user_orig = idx2user[u]
            rec_items = [idx2item[i] for i in topk]

            # one row: user_id followed by K item_ids
            results.append([user_orig] + rec_items)

    # uses your existing helper from cell 2
    write_results_to_file(results, out_path)
    print(f"Saved submission to {out_path}")


if __name__ == "__main__":

    # FIX 6 (Part 1): Instantiate args object first
    args = NeuralMFArgs()

    ui_pos, user2idx, idx2user, item2idx, idx2item = build_index()

    train_ui, valid_ui = train_valid_split(ui_pos, valid_ratio=args.valid_ratio, seed=args.seed)

    num_users = len(user2idx)
    num_items = len(item2idx)

    # popularity-weighted negatives help stability
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
    model = NeuralMF(num_users, num_items, f=args.emb_mf, mlp_dim=args.emb_mlp).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    bce_loss = nn.BCEWithLogitsLoss()

    best_ndcg, best_path = 0.0, None
    for epoch in range(1, args.epochs + 1):
        print(f"Running epoch {epoch}")
        # FIX 6 (Part 3): Pass device argument to train_epoch
        loss = train_epoch(model, loader, optimizer, device, bce_loss)
        n20 = evaluate(model, train_ui, valid_ui, K=args.topk, device=device)
        print(f"Epoch {epoch:02d} | loss={loss:.4f} | NDCG@{args.topk}={n20:.4f}")

        if n20 > best_ndcg:
            best_ndcg = n20
            best_path = os.path.join(args.outdir, "neumf_best.pt")
            os.makedirs(args.outdir, exist_ok=True)
            torch.save({"model": model.state_dict(),
                        "user2idx": user2idx,
                        "item2idx": item2idx,
                        "idx2user": idx2user,
                        "idx2item": idx2item}, best_path)

    # load best (optional) and write submission on train+valid seen filter
    if best_path:
        ckpt = torch.load(best_path, map_location=device)
        model.load_state_dict(ckpt["model"])

    # Merge seen sets from train+valid to avoid recommending any seen items
    # The original code only defines seen_all but does not use it for final prediction/submission
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
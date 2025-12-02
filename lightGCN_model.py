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
import scipy.sparse as sp

@dataclass
class LightGCNArgs:
    valid_ratio = 0.1
    seed = 42
    num_neg = 1
    batch_size = 4096
    epochs = 20
    lr = 1e-3
    
    emb_dim = 64
    n_layers = 3
    
    topk = 20
    reg_weight = 1e-4
    
    # NOTE: These settings worked well on my M4 Pro setup
    use_mps = True
    num_workers = 0  # Had issues with multiprocessing on macOS
    compile_model = True
    outdir = "./checkpoints"
    
    # Different variations I want to try
    use_tweak1 = False  # Make it deeper and wider
    use_tweak2 = False  # Add some learning rate decay


class LightGCN(nn.Module):
    def __init__(self, num_users, num_items, emb_dim=64, n_layers=3, adj_mat=None):
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items  
        self.emb_dim = emb_dim
        self.n_layers = n_layers
        
        # Create the embedding layers
        self.user_embedding = nn.Embedding(num_users, emb_dim)
        self.item_embedding = nn.Embedding(num_items, emb_dim)
        
        # Xavier initialization - found this works better than random init
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_embedding.weight)
        
        # Keep the adjacency matrix
        self.Graph = adj_mat
        
    def compute_graph_embeddings(self):
        # Get initial embeddings
        users_emb = self.user_embedding.weight
        items_emb = self.item_embedding.weight
        all_emb = torch.cat([users_emb, items_emb], dim=0)
        
        # Store all layer outputs
        embs = [all_emb]
        
        # Do the graph convolution - unfortunately MPS doesn't support sparse ops well
        # so we have to move back and forth between CPU and GPU
        device = all_emb.device
        for layer_idx in range(self.n_layers):
            # Move to CPU for sparse multiplication
            all_emb_cpu = all_emb.cpu()
            all_emb_cpu = torch.sparse.mm(self.Graph.cpu(), all_emb_cpu)
            # Move back to MPS
            all_emb = all_emb_cpu.to(device)
            embs.append(all_emb)
        
        # Average all the embeddings from different layers
        embs = torch.stack(embs, dim=1)
        light_out = torch.mean(embs, dim=1)
        
        # Split back into users and items
        users_emb_final, items_emb_final = torch.split(
            light_out, [self.num_users, self.num_items]
        )
        
        return users_emb_final, items_emb_final
    
    def forward(self, users, pos_items, neg_items):
        # Get the final embeddings after graph convolution
        all_users_emb, all_items_emb = self.compute_graph_embeddings()
        
        # Extract embeddings for the batch
        users_emb = all_users_emb[users]
        pos_emb = all_items_emb[pos_items]
        neg_emb = all_items_emb[neg_items]
        
        # Calculate scores - need to handle multiple negatives
        if neg_emb.dim() == 3:  # Multiple negative items per user
            users_emb = users_emb.unsqueeze(1)  # Broadcasting trick
            pos_scores = torch.sum(users_emb * pos_emb.unsqueeze(1), dim=-1)
            neg_scores = torch.sum(users_emb * neg_emb, dim=-1)
        else:
            # Simple dot product for scores
            pos_scores = torch.sum(users_emb * pos_emb, dim=1)
            neg_scores = torch.sum(users_emb * neg_emb, dim=1)
        
        return pos_scores, neg_scores
    
    @torch.no_grad()
    def predict_users(self, user_ids):
        # Get embeddings for prediction
        all_users_emb, all_items_emb = self.compute_graph_embeddings()
        users_emb = all_users_emb[user_ids]
    
        # Calculate scores for all items
        scores = torch.matmul(users_emb, all_items_emb.t())
        return scores


def create_adjacency_matrix(train_ui, num_users, num_items, device):
    # Build the user-item interaction lists
    row_indices, col_indices = [], []
    for user_id, items in train_ui.items():
        for item_id in items:
            row_indices.append(user_id)
            col_indices.append(item_id)
    
    # Convert to numpy for efficiency
    row_indices = np.array(row_indices, dtype=np.int32)
    col_indices = np.array(col_indices, dtype=np.int32)
    data_values = np.ones(len(row_indices), dtype=np.float32)
    
    # Create the user-item matrix
    R = sp.coo_matrix((data_values, (row_indices, col_indices)), shape=(num_users, num_items))
    
    # Build the full adjacency matrix (users + items)
    total_nodes = num_users + num_items
    adj_mat = sp.dok_matrix((total_nodes, total_nodes), dtype=np.float32)
    adj_mat = adj_mat.tolil()  # Convert for efficient assignment
    
    R_csr = R.tocsr()
    adj_mat[:num_users, num_users:] = R_csr  # User-item connections
    adj_mat[num_users:, :num_users] = R_csr.T  # Item-user connections
    adj_mat = adj_mat.tocoo()
    
    # Normalize the adjacency matrix (symmetric normalization)
    degree_sum = np.array(adj_mat.sum(axis=1)).flatten()
    d_inv_sqrt = np.power(degree_sum, -0.5)
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.  # Handle isolated nodes
    
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    norm_adj = d_mat_inv_sqrt.dot(adj_mat).dot(d_mat_inv_sqrt).tocoo()
    
    # Convert to PyTorch sparse tensor - keep on CPU because MPS doesn't like sparse ops
    indices = torch.LongTensor(np.vstack([norm_adj.row, norm_adj.col]))
    values = torch.FloatTensor(norm_adj.data)
    shape = torch.Size(norm_adj.shape)
    
    sparse_tensor = torch.sparse_coo_tensor(indices, values, shape, dtype=torch.float32)
    return sparse_tensor  # Keep on CPU


# I tried torch.compile on this but it didn't speed things up much
@torch.compile(mode="default", fullgraph=False)
def bpr_loss_compiled(pos_scores, neg_scores):
    return -torch.mean(torch.log(torch.sigmoid(pos_scores - neg_scores) + 1e-10))


def train_epoch(model, loader, optimizer, device, reg_weight, use_compile=True):
    model.train()
    total_loss_val = 0.0
    total_bpr_val = 0.0
    total_reg_val = 0.0
    
    # Training loop
    for batch_data in tqdm(loader, desc="Training Progress", leave=False):
        users, pos_items, neg_items = batch_data
        users = users.long().to(device, non_blocking=True)
        pos_items = pos_items.long().to(device, non_blocking=True)
        neg_items = neg_items.long().to(device, non_blocking=True)
        
        # Handle case where we have multiple negatives per user
        if neg_items.dim() == 2:
            batch_size, num_negs = neg_items.shape
        else:
            batch_size = neg_items.shape[0]
            num_negs = 1
            neg_items = neg_items.unsqueeze(1)
        
        # Forward pass through model
        pos_scores, neg_scores = model(users, pos_items, neg_items)
        
        # Compute BPR loss - handle multiple negatives case
        if neg_scores.dim() == 2:
            # Multiple negatives: compute loss for each negative
            bpr_loss = -torch.mean(torch.log(torch.sigmoid(pos_scores.unsqueeze(1) - neg_scores) + 1e-10))
        else:
            # Single negative case
            bpr_loss = -torch.mean(torch.log(torch.sigmoid(pos_scores - neg_scores) + 1e-10))
        
        # Regularization on the original embeddings (before graph convolution)
        user_emb_0 = model.user_embedding(users)
        pos_item_emb_0 = model.item_embedding(pos_items)
        neg_items_flat = neg_items.view(-1)
        neg_item_emb_0 = model.item_embedding(neg_items_flat)
        
        # L2 regularization term
        reg_loss = reg_weight * (
            torch.norm(user_emb_0) ** 2 + 
            torch.norm(pos_item_emb_0) ** 2 + 
            torch.norm(neg_item_emb_0) ** 2
        ) / (2 * users.shape[0])
        
        # Total loss
        total_loss = bpr_loss + reg_loss
        
        # Backpropagation
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        
        # Track losses
        total_loss_val += total_loss.item()
        total_bpr_val += bpr_loss.item()
        total_reg_val += reg_loss.item()
    
    # Calculate averages
    num_batches = max(1, len(loader))
    return total_loss_val / num_batches, total_bpr_val / num_batches, total_reg_val / num_batches


@torch.no_grad()
def evaluate(model, train_ui, valid_ui, K=20, batch_users=1024, device="cpu", max_eval_users=5000):
    model.eval()
    
    # Get users that have validation data
    eval_users = [u for u, items in valid_ui.items() if len(items) > 0]
    if not eval_users:
        return 0.0
    
    # Limit evaluation size for speed during development
    if len(eval_users) > max_eval_users:
        eval_users = random.sample(eval_users, max_eval_users)
    
    ndcg_scores = []
    
    # Process users in batches
    for start_idx in tqdm(range(0, len(eval_users), batch_users), desc="Evaluate Progress", leave=False):
        batch_users_list = eval_users[start_idx:start_idx + batch_users]
        user_tensor = torch.tensor(batch_users_list, dtype=torch.long, device=device)
        
        # Get predictions for this batch
        scores = model.predict_users(user_tensor)
        scores = scores.cpu().numpy()
        
        # Process each user in the batch
        for row_idx, user_id in enumerate(batch_users_list):
            # Mask out items the user has already seen
            seen_items = train_ui.get(user_id, set())
            if seen_items:
                seen_list = list(seen_items)
                scores[row_idx, seen_list] = -1e9
            
            # Get top-K recommendations efficiently
            if K < len(scores[row_idx]):
                top_k_items = np.argpartition(-scores[row_idx], K)[:K]
                top_k_items = top_k_items[np.argsort(-scores[row_idx][top_k_items])]
            else:
                top_k_items = np.argsort(-scores[row_idx])[:K]
            
            # Calculate NDCG
            ndcg_score = ndcg_at_k(top_k_items.tolist(), valid_ui[user_id], K)
            ndcg_scores.append(ndcg_score)
    
    return float(np.mean(ndcg_scores))


@torch.no_grad()
def generate_submission_fast(
    model,
    seen_ui,
    num_users,
    idx2user,
    idx2item,
    device,
    K=20,
    user_batch_size=1024,
    out_path="output.txt",
):
    model.eval()
    all_user_indices = list(range(num_users))
    results = []
    
    # Process all users in batches
    for start_idx in tqdm(range(0, num_users, user_batch_size), desc="Predict Progress"):
        batch_users = all_user_indices[start_idx:start_idx + user_batch_size]
        user_tensor = torch.tensor(batch_users, dtype=torch.long, device=device)
        
        # Get predictions
        scores = model.predict_users(user_tensor)
        scores = scores.cpu().numpy()
        
        # Process each user
        for row_idx, user_idx in enumerate(batch_users):
            # Mask seen items
            seen_items = seen_ui.get(user_idx, set())
            if seen_items:
                scores[row_idx, list(seen_items)] = -1e9
            
            # Get top-K efficiently
            top_k_items = np.argpartition(-scores[row_idx], K)[:K]
            top_k_items = top_k_items[np.argsort(-scores[row_idx][top_k_items])]
            
            # Convert back to original IDs
            original_user = idx2user[user_idx]
            recommended_items = [idx2item[item_idx] for item_idx in top_k_items]
            
            results.append([original_user] + recommended_items)
    
    # Save results
    write_results_to_file(results, out_path)
    print(f" Saved submission to {out_path}")


def run_experiment(args, experiment_name="base"):
    print(f"\n{'='*70}")
    print(f" Running Experiment: {experiment_name}")
    print(f"{'='*70}")
    print(f"Config: emb_dim={args.emb_dim}, n_layers={args.n_layers}, lr={args.lr}")
    print(f"{'='*70}\n")
    
    # Set random seeds for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    
    # Device setup - prefer MPS on Mac
    if args.use_mps and torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using Apple M4 Pro GPU (Metal Performance Shaders)")
    else:
        device = torch.device("cpu")
        print("Using CPU")
    
    # Data loading
    print("\n Loading dataset...")
    ui_pos, user2idx, idx2user, item2idx, idx2item = build_index()
    train_ui, valid_ui = train_valid_split(ui_pos, valid_ratio=args.valid_ratio, seed=args.seed)
    
    num_users = len(user2idx)
    num_items = len(item2idx)
    
    print(f" Users: {num_users:,} | Items: {num_items:,}")
    
    # Create training dataset
    dataset = PairDataset(train_ui, num_items, num_neg=args.num_neg, pop_weights=None)
    
    train_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=args.num_workers,
        pin_memory=False,
        persistent_workers=False
    )
    
    # Build the graph
    print("\n Building graph adjacency matrix...")
    adj_matrix = create_adjacency_matrix(train_ui, num_users, num_items, device)
    print(f" Graph shape: {adj_matrix.shape}")
    
    # Initialize the model
    print("\n Initializing LightGCN...")
    model = LightGCN(
        num_users=num_users,
        num_items=num_items,
        emb_dim=args.emb_dim,
        n_layers=args.n_layers,
        adj_mat=adj_matrix
    ).to(device)
    
    # Try to compile the model - this sometimes helps on newer PyTorch versions
    if args.compile_model and hasattr(torch, 'compile'):
        print("⚡ Trying to compile model with torch.compile...")
        try:
            model = torch.compile(model, mode="default")
            print(" Model compiled successfully")
        except Exception as e:
            print(f"  Compilation failed: {e}, using eager mode")
    
    # Setup optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    
    # Optional learning rate scheduler
    scheduler = None
    if args.use_tweak2:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
        print(" Using Cosine Annealing LR Scheduler")
    
    # Training variables
    best_ndcg_score = 0.0
    best_model_path = None
    
    print(f"\n{'='*70}")
    print(f"  Starting Training - {args.epochs} Epochs")
    print(f"{'='*70}\n")
    
    # Training loop
    for epoch in range(1, args.epochs + 1):
        # Train for one epoch
        train_loss, bpr_loss, reg_loss = train_epoch(
            model, train_loader, optimizer, device, args.reg_weight, use_compile=args.compile_model
        )
        
        # Update learning rate if using scheduler
        if scheduler is not None:
            current_lr = scheduler.get_last_lr()[0]
            scheduler.step()
        else:
            current_lr = args.lr
        
        # Evaluate on validation set
        ndcg_score = evaluate(model, train_ui, valid_ui, K=args.topk, device=device)
        
        # Print progress
        print(f"Epoch {epoch:2d}/{args.epochs} | Loss: {train_loss:.4f} (BPR: {bpr_loss:.4f}, Reg: {reg_loss:.4f}) | LR: {current_lr:.6f} | NDCG@{args.topk}: {ndcg_score:.4f}", end="")
        
        # Save best model
        if ndcg_score > best_ndcg_score:
            best_ndcg_score = ndcg_score
            best_model_path = os.path.join(args.outdir, f"lightgcn_{experiment_name}_best.pt")
            os.makedirs(args.outdir, exist_ok=True)
            
            # Handle compiled models - need to save the original
            model_to_save = model._orig_mod if hasattr(model, '_orig_mod') else model
            
            torch.save({
                "model": model_to_save.state_dict(),
                "user2idx": user2idx,
                "item2idx": item2idx,
                "idx2user": idx2user,
                "idx2item": idx2item,
                "args": args,
                "experiment": experiment_name
            }, best_model_path)
            print(" NEW BEST!")
        else:
            print()
    
    print(f"\n{'='*70}")
    print(f" {experiment_name} Training Complete!")
    print(f" Best NDCG@{args.topk}: {best_ndcg_score:.4f}")
    print(f"{'='*70}\n")
    
    # Load the best model for final predictions
    if best_model_path:
        print(" Loading best model for final predictions...")
        checkpoint = torch.load(best_model_path, map_location=device, weights_only=False)
        
        # Load state dict appropriately
        if args.compile_model and hasattr(model, '_orig_mod'):
            model._orig_mod.load_state_dict(checkpoint["model"])
        else:
            model.load_state_dict(checkpoint["model"])
    
    # Generate final recommendations
    print("\n Generating final recommendations...")
    # Combine train and validation sets as "seen" items
    seen_all_items = {u: set(train_ui.get(u, set())) | set(valid_ui.get(u, set())) 
                     for u in range(num_users)}
    
    output_filename = f"output_{experiment_name}.txt"
    generate_submission_fast(
        model=model,
        seen_ui=seen_all_items,
        num_users=num_users,
        idx2user=idx2user,
        idx2item=idx2item,
        device=device,
        K=args.topk,
        user_batch_size=1024,
        out_path=output_filename,
    )
    
    return best_ndcg_score, output_filename


if __name__ == "__main__":
    
    # EXPERIMENT 1: BASE MODEL - starting with the basic setup
    print("\n" + "="*70)
    print(" EXPERIMENT 1: BASE MODEL")
    print("="*70)
    
    base_args = LightGCNArgs()
    base_ndcg, base_file = run_experiment(base_args, experiment_name="base")
    
    
    # EXPERIMENT 2: TWEAK 1 - trying by making it bigger and deeper
    print("\n" + "="*70)
    print(" EXPERIMENT 2: TWEAK 1 - DEEPER + WIDER")
    print("="*70)
    
    tweak1_args = LightGCNArgs()
    tweak1_args.use_tweak1 = True
    tweak1_args.emb_dim = 128  # Make it wider
    tweak1_args.n_layers = 4   # Make it deeper
    tweak1_args.lr = 3e-4      # Lower learning rate for stability
    tweak1_args.num_neg = 4    # More negative samples
    tweak1_args.reg_weight = 1e-5  # Less regularization
    
    tweak1_ndcg, tweak1_file = run_experiment(tweak1_args, experiment_name="tweak1_deeper_wider")
    
    
    # EXPERIMENT 3: TWEAK 2 - trying some learning rate scheduling
    print("\n" + "="*70)
    print(" EXPERIMENT 3: TWEAK 2 - LR SCHEDULING")
    print("="*70)
    
    tweak2_args = LightGCNArgs()
    tweak2_args.use_tweak2 = True
    tweak2_args.lr = 1e-3  # Start higher since we'll decay
    tweak2_args.num_neg = 4  # More negatives
    tweak2_args.reg_weight = 1e-5
    
    tweak2_ndcg, tweak2_file = run_experiment(tweak2_args, experiment_name="tweak2_lr_schedule")
    
    # FINAL SUMMARY
    print("\n" + "="*70)
    print(" FINAL RESULTS SUMMARY")
    print("="*70)
    print(f"\n{'Experiment':<30} {'NDCG@20':<12} {'Output File':<30}")
    print("-" * 70)
    print(f"{'Base Model':<30} {base_ndcg:<12.4f} {base_file:<30}")
    print(f"{'Tweak 1: Deeper+Wider':<30} {tweak1_ndcg:<12.4f} {tweak1_file:<30}")
    print(f"{'Tweak 2: LR Scheduling':<30} {tweak2_ndcg:<12.4f} {tweak2_file:<30}")
    print("-" * 70)
    
    # Figure out which experiment did best
    all_results = [
        ("Base Model", base_ndcg, base_file),
        ("Tweak 1: Deeper+Wider", tweak1_ndcg, tweak1_file),
        ("Tweak 2: LR Scheduling", tweak2_ndcg, tweak2_file)
    ]
    best_result = max(all_results, key=lambda x: x[1])
    
    print(f"\n BEST MODEL: {best_result[0]}")
    print(f"    NDCG@20: {best_result[1]:.4f}")
    print(f"    Output: {best_result[2]}")
    print(f"\n Use '{best_result[2]}' for your final submission!")
    
    # Copy the best result to the standard output file
    import shutil
    final_output_path = "output.txt"
    shutil.copy(best_result[2], final_output_path)
    print(f"\n Best predictions copied to '{final_output_path}'")
    
    print("\n" + "="*70 + "\n")
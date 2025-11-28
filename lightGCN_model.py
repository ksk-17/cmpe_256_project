import torch
import torch.nn as nn
import torch.optim as optim
from scipy.sparse import csr_matrix
import numpy as np
from tqdm import tqdm
import time

from process_data import load_dataset, get_statistics_and_csr, split_csr_train_val_test, convert_csr_to_edge_list
from utils import ndcg_at_k_from_factors, write_results_to_file


class LightGCN(nn.Module):
    def __init__(self, n_users, n_items, embedding_dim, n_layers=3):
        super(LightGCN, self).__init__()
        
        self.n_users = n_users
        self.n_items = n_items
        self.n_layers = n_layers
        self.embedding_dim = embedding_dim
        
        self.user_embeddings = nn.Embedding(n_users, embedding_dim)
        self.item_embeddings = nn.Embedding(n_items, embedding_dim)
        
        nn.init.normal_(self.user_embeddings.weight, mean=0.0, std=0.01)
        nn.init.normal_(self.item_embeddings.weight, mean=0.0, std=0.01)

    def forward(self, adj_matrix_norm):
        # initial embeddings E^(0)
        E0 = torch.cat([self.user_embeddings.weight, self.item_embeddings.weight], dim=0)
        all_embeddings = [E0]
        E = E0
        
        # Propagation Loop (K layers)
        for k in range(self.n_layers):
            E = torch.sparse.mm(adj_matrix_norm, E)
            all_embeddings.append(E)

        # Final Embedding (Aggregation)
        # Average the embeddings from all layers: E* = sum(E^(k)) / (K+1)
        all_embeddings = torch.stack(all_embeddings, dim=1)
        final_embeddings = torch.mean(all_embeddings, dim=1)
        
        # Split back into User and Item factors
        user_final_factors = final_embeddings[:self.n_users]
        item_final_factors = final_embeddings[self.n_users:]
        
        return user_final_factors, item_final_factors

    def get_user_item_embeddings(self):
        """Helper to get user/item embeddings without graph propagation."""
        return self.user_embeddings.weight, self.item_embeddings.weight


def bpr_loss(users, pos_items, neg_items, model: LightGCN, reg_weight=1e-4):
    """
    Bayesian Personalized Ranking (BPR) Loss function for implicit feedback.
    :param users, pos_items, neg_items: Tensors of user, positive item, and negative item indices.
    """
    
    # initial embeddings (E0) for regularization term
    user_embeds_e0, item_embeds_e0 = model.get_user_item_embeddings()
    
    # initial embeddings for the current batch
    u_e = user_embeds_e0[users]
    p_i_e = item_embeds_e0[pos_items]
    n_i_e = item_embeds_e0[neg_items]

    # Get final propagated embeddings (E*)
    user_final, item_final = model.final_factors
    u_f = user_final[users]
    p_i_f = item_final[pos_items]
    n_i_f = item_final[neg_items]

    # Calculate scores (predictions)
    # Score(user, positive) = dot(u_f, p_i_f)
    pos_scores = torch.sum(u_f * p_i_f, dim=1)
    # Score(user, negative) = dot(u_f, n_i_f)
    neg_scores = torch.sum(u_f * n_i_f, dim=1)
    
    # BPR loss: -ln(sigmoid(pos_score - neg_score))
    log_sigmoid = nn.LogSigmoid()
    bpr_loss = -torch.mean(log_sigmoid(pos_scores - neg_scores))

    # Add L2 Regularization term (on E0 embeddings)
    # This is calculated on the initial embeddings of the batch items/users
    l2_reg = reg_weight * (torch.sum(u_e.pow(2)) + torch.sum(p_i_e.pow(2)) + torch.sum(n_i_e.pow(2)))

    return bpr_loss + l2_reg

def get_adjacency_matrix(n_users, n_items, train_csr: csr_matrix):
    """Creates the normalized adjacency matrix A_norm."""
    # full adjacency matrix (N+M) x (N+M) for the graph
    n_nodes = n_users + n_items
    
    # COO format for the train data
    coo = train_csr.tocoo()
    
    # indices for User-Item and Item-User edges
    u_idx, i_idx = coo.row, coo.col
    u_indices = torch.tensor(u_idx, dtype=torch.long)
    i_indices = torch.tensor(i_idx + n_users, dtype=torch.long)
    
    # Adjacency indices (row, col)
    # [User -> Item] and [Item -> User]
    indices = torch.stack([
        torch.cat([u_indices, i_indices]),  # Row index
        torch.cat([i_indices, u_indices])   # Col index
    ], dim=0)
    
    values = torch.ones(indices.size(1))
    
    # Create the sparse matrix A
    A = torch.sparse_coo_tensor(indices, values, (n_nodes, n_nodes)).coalesce()

    # Degree Matrix D^(-1/2) for normalization
    degrees_sparse = torch.sparse.sum(A, dim=1)
    
    # degrees_sparse to a dense tensor for easy indexing
    degrees_dense = degrees_sparse.to_dense()

    # Calculate D^(-1/2) for all nodes using torch.pow (safer than tensor.pow())
    D_inv_sqrt = torch.pow(degrees_dense, -0.5)
    # Handle isolated nodes (where degree is 0, resulting in Inf power)
    D_inv_sqrt[torch.isinf(D_inv_sqrt)] = 0.

    # A_norm = D^(-1/2) * A * D^(-1/2)
    rows, cols = A.indices()[0], A.indices()[1]
    values = A.values() 

    # The normalization factor for edge (r, c) is D_inv_sqrt[r] * D_inv_sqrt[c]
    norm_values = D_inv_sqrt[rows] * values * D_inv_sqrt[cols]
    
    A_norm = torch.sparse_coo_tensor(indices, norm_values, (n_nodes, n_nodes)).coalesce()
    
    return A_norm


def sample_negative(users, pos_items, n_items, train_csr: csr_matrix):
    """
    Samples a negative item (not interacted with by the user) for each positive user-item pair.
    """
    neg_items = []
    train_csr = train_csr.tocsr() # CSR for efficient lookup
    
    for u, i in zip(users.numpy(), pos_items.numpy()):
        while True:
            # Sample a random item index
            neg_item = np.random.randint(n_items)
            
            # Check if it's a non-interaction (not in train_csr[u])
            # CSR matrix lookup: R[u, i] != 0 means interaction
            if train_csr[u, neg_item] == 0:
                neg_items.append(neg_item)
                break
                
    return torch.tensor(neg_items, dtype=torch.long)


def run_experiment(exp_name, n_layers, embed_dim, n_epochs, train_csr, val_csr, adj_matrix_norm, n_users, n_items, train_users, train_items):
    print(f"\n Running Experiment: {exp_name}")
    print(f"Configuration: Layers={n_layers}, Embedding Dim={embed_dim}")
    
    # Initializing Model with specific Tweak parameters
    model = LightGCN(n_users, n_items, embed_dim, n_layers)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    n_interactions = len(train_users)
    best_ndcg = 0.0
    BATCH_SIZE = 2048
    
    for epoch in range(1, n_epochs + 1):
        model.train()
        start_time = time.time()
        
        indices = torch.randperm(n_interactions)
        total_loss = 0.0
        
        # Training Batch Loop
        for i in tqdm(range(0, n_interactions, BATCH_SIZE), desc=f"[{exp_name}] Epoch {epoch}"):
            batch_indices = indices[i:i + BATCH_SIZE]
            batch_users = train_users[batch_indices]
            batch_pos_items = train_items[batch_indices]
            
            batch_neg_items = sample_negative(batch_users, batch_pos_items, n_items, train_csr)

            optimizer.zero_grad()
            user_final, item_final = model(adj_matrix_norm)
            model.final_factors = (user_final, item_final)
            
            loss = bpr_loss(batch_users, batch_pos_items, batch_neg_items, model)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        # Validation
        model.eval()
        with torch.no_grad():
            user_final, item_final = model(adj_matrix_norm)
            user_final_np = user_final.cpu().numpy()
            item_final_np = item_final.cpu().numpy()
            
            val_ndcg = ndcg_at_k_from_factors(R_test=val_csr, X=user_final_np, Y=item_final_np, R_train=train_csr, k=20)
            
            if val_ndcg > best_ndcg:
                best_ndcg = val_ndcg

    print(f"Finished {exp_name}. Best Validation NDCG: {best_ndcg:.4f}")
    return best_ndcg, model


if __name__ == "__main__":
    print(" 1. Data Loading ")
    df = load_dataset()
    csr, idx_to_user, idx_to_item = get_statistics_and_csr(df)
    train_csr, val_csr, test_csr = split_csr_train_val_test(csr, 0.1, 0.1)

    n_users, n_items = csr.shape
    train_users, train_items = convert_csr_to_edge_list(train_csr)
    
    # Pre-calculate Graph 
    print(" 2. Building Graph ")
    adj_matrix_norm = get_adjacency_matrix(n_users, n_items, train_csr)

    # Tweaks
    # I'm running the model 3 times with different settings
    experiments = [
        ("Baseline", 3, 64),         # Standard: 3 Layers, 64 Dim
        ("Tweak 1: 1 Layer", 1, 64), # Tweak1
        ("Tweak 2: 32 Dim", 3, 32)   # Tweak2
    ]
    
    results = {}
    best_model_overall = None
    best_ndcg_overall = 0.0
    
    # Experiments
    for name, layers, dim in experiments:
        ndcg, trained_model = run_experiment(
            name, layers, dim, 
            n_epochs=5, 
            train_csr=train_csr, val_csr=val_csr, 
            adj_matrix_norm=adj_matrix_norm, 
            n_users=n_users, n_items=n_items, 
            train_users=train_users, train_items=train_items
        )
        results[name] = ndcg
        
        if ndcg > best_ndcg_overall:
            best_ndcg_overall = ndcg
            best_model_overall = trained_model

    #  Comparison Table
    print("FINAL EXPERIMENT RESULTS \n")
    for name, score in results.items():
        print(f"{name}: NDCG@20 = {score:.4f}")
        
    #  Recommendations 
    print(f"\nGenerating recommendations using the best hyperparameters...")
    
    best_model_overall.eval()
    with torch.no_grad():
        user_final, item_final = best_model_overall(adj_matrix_norm)
        user_final_np = user_final.cpu().numpy()
        item_final_np = item_final.cpu().numpy()
        
        # Final Test Score
        test_ndcg = ndcg_at_k_from_factors(R_test=test_csr, X=user_final_np, Y=item_final_np, R_train=train_csr, k=20)
        print(f"Final TEST NDCG@20 (Unseen Data): {test_ndcg:.4f}")

        output_results = []
        for u in tqdm(range(n_users), desc="Writing Predictions"):
            scores = user_final_np[u] @ item_final_np.T
            
            start, end = train_csr.indptr[u], train_csr.indptr[u + 1]
            seen = train_csr.indices[start: end]
            
            if len(seen) > 0:
                scores[seen] = -np.inf

            topk = np.argsort(-scores)[:20]

            user_original_id = idx_to_user.get(u, u)
            item_original_ids = [idx_to_item.get(item_id, item_id) for item_id in topk]
            
            output_results.append([user_original_id, *item_original_ids])

        write_results_to_file(output_results, "lightgcn_best_output.txt")
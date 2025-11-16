import numpy as np
from scipy.sparse import csr_matrix
from tqdm import tqdm

# NDCG Function
def ndcg_at_k_from_factors(R_test: csr_matrix, X: np.ndarray, Y: np.ndarray, R_train: csr_matrix=None, k=10) -> float:
    print(f"Calculating NDCG@{k}")
    n_users = R_test.shape[0]
    ndcgs = []

    for u in tqdm(range(n_users), desc="Evaluation Progress"):
        start_t, end_t = R_test.indptr[u], R_test.indptr[u+1]
        gt_items = R_test.indices[start_t:end_t]
        if len(gt_items) == 0:
            continue

        scores = X[u] @ Y.T
        if R_train is not None:
            start_tr, end_tr = R_train.indptr[u], R_train.indptr[u+1]
            seen_items = R_train.indices[start_tr:end_tr]
            if len(seen_items) > 0:
                scores[seen_items] = -np.inf

        # get top-k
        if k >= len(scores):
            topk_idx = np.argsort(-scores)
        else:
            idx = np.argpartition(-scores, k)[:k]
            topk_idx = idx[np.argsort(-scores[idx])]

        rel = np.isin(topk_idx, gt_items).astype(np.float32)

        # DCG@k
        dcg = np.sum(rel / np.log2(np.arange(2, len(rel) + 2)))

        # Ideal DCG
        idcg = np.sum(1.0 / np.log2(np.arange(2, min(len(gt_items), k) + 2)))

        ndcg = dcg / idcg if idcg > 0 else 0.0
        ndcgs.append(ndcg)
    
    return np.mean(ndcgs) if ndcgs else 0.0


def write_results_to_file(results: list[list], path: str):
    with open(path, "w") as f:
        for row in results:
            line = " ".join(map(str, row))
            f.write(line + "\n")
    print(f"Results written to {path} ({len(results)} rows)")

        
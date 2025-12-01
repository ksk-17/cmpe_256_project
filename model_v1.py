import numpy as np
from scipy.sparse import csr_matrix
from tqdm import tqdm
from utils import ndcg_at_k_from_factors, write_results_to_file
from process_data import (
    load_dataset,
    get_statistics_and_csr,
    split_csr_train_val_test,
)

class ImplicitMF:
    def __init__(self, n_factors=64, alpha=40.0, reg=0.1, n_iters=32, seed=42):
        self.f = n_factors
        self.alpha = alpha
        self.reg = reg
        self.n_iters = n_iters
        self.rng = np.random.default_rng(seed)
        self.user_factors = None
        self.item_factors = None

    def als_user_step(self, R: csr_matrix):
        Y = self.item_factors
        YTY = Y.T @ Y
        X = self.user_factors

        indptr, indices = R.indptr, R.indices
        for u in range(R.shape[0]):
            start, end = indptr[u], indptr[u + 1]
            Iu = indices[start:end]
            if len(Iu) == 0:
                continue
            YI = Y[Iu]

            Cu = 1.0 + self.alpha * np.ones(len(Iu))
            Cu_minus_I = Cu - 1.0

            A = YTY + (YI.T * Cu_minus_I) @ YI + self.reg * np.eye(self.f)
            b = (YI * Cu[:, None]).sum(axis=0)
            X[u] = np.linalg.solve(A, b)

        self.user_factors = X

    def als_item_step(self, R: csr_matrix):
        X = self.user_factors
        XTX = X.T @ X
        Y = self.item_factors

        RT = R.T.tocsr()
        indptr, indices = RT.indptr, RT.indices
        for i in range(R.shape[1]):
            start, end = indptr[i], indptr[i + 1]
            Ui = indices[start:end]
            if len(Ui) == 0:
                continue
            XU = X[Ui]

            Cu = 1.0 + self.alpha * np.ones(len(Ui))
            Cu_minus_I = Cu - 1.0

            A = XTX + (XU.T * Cu_minus_I) @ XU + self.reg * np.eye(self.f)
            b = (XU * Cu[:, None]).sum(axis=0)
            Y[i] = np.linalg.solve(A, b)

        self.item_factors = Y

    def fit(self, R: csr_matrix):
        n_users, n_items = R.shape
        self.user_factors = 0.01 * self.rng.standard_normal((n_users, self.f))
        self.item_factors = 0.01 * self.rng.standard_normal((n_items, self.f))

        for _ in tqdm(range(1, self.n_iters + 1), desc="ALS Progress"):
            self.als_user_step(R)
            self.als_item_step(R)

        return self

    def predict_user(self, user_id: int):
        return self.user_factors[user_id] @ self.item_factors.T

    def recommend(self, R: csr_matrix, user_id: int,
                  user_idx_to_id: dict, item_idx_to_id: dict, k=20):
        scores = self.predict_user(user_id)
        start, end = R.indptr[user_id], R.indptr[user_id + 1]
        seen = set(R.indices[start:end])
        if seen:
            scores = scores.copy()
            scores[list(seen)] = -np.inf

        if k >= len(scores):
            topk = np.argsort(-scores)
        else:
            idx = np.argpartition(-scores, k)[:k]
            topk = idx[np.argsort(-scores[idx])]

        user_original_id = user_idx_to_id.get(user_id, user_id)
        item_original_ids = [
            item_idx_to_id.get(item_id, item_id) for item_id in topk[:k]
        ]
        return user_original_id, item_original_ids, scores[topk]


if __name__ == "__main__":
    df = load_dataset()
    csr_mat, idx_to_user, idx_to_item = get_statistics_and_csr(df)

    train_csr, val_csr, test_csr = split_csr_train_val_test(csr_mat, 0.1, 0.1)

    print("\n=== Training MF on train_csr for evaluation ===")
    mf_eval = ImplicitMF(n_factors=64, alpha=40.0, reg=0.1, n_iters=32)
    mf_eval.fit(train_csr)

    val_ndcg = ndcg_at_k_from_factors(
        R_test=val_csr,
        X=mf_eval.user_factors,
        Y=mf_eval.item_factors,
        R_train=train_csr,
        k=20,
    )
    print(f"Val NDCG@20: {val_ndcg:.4f}")

    test_ndcg = ndcg_at_k_from_factors(
        R_test=test_csr,
        X=mf_eval.user_factors,
        Y=mf_eval.item_factors,
        R_train=train_csr,
        k=20,
    )
    print(f"Test NDCG@20: {test_ndcg:.4f}")
 
    print("\n=== Re-fitting MF on FULL csr_mat for final submission (strong config) ===")
    mf_full = ImplicitMF(
    n_factors=128,  
    alpha=40.0,
    reg=0.07,        
    n_iters=26      
    )
               
    mf_full.fit(csr_mat)


    results = []
    for u in tqdm(range(csr_mat.shape[0]), desc="Predictions Progress"):
        user, items, _ = mf_full.recommend(
            csr_mat,  
            user_id=u,
            user_idx_to_id=idx_to_user,
            item_idx_to_id=idx_to_item,
            k=20,
        )
        results.append([user, *items])

    write_results_to_file(results, "output_mf_full.txt")

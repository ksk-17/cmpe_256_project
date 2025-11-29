import numpy as np
from scipy.sparse import csr_matrix
from tqdm import tqdm
from utils import ndcg_at_k_from_factors, write_results_to_file
from process_data import load_dataset, get_statistics_and_csr, split_csr_train_val_test

class ImplicitMF:
    def __init__(self, n_factors=64, alpha=40.0, reg=0.1, n_iters=15, seed=42):
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
            start, end = indptr[u], indptr[u+1]
            Iu = indices[start:end]
            YI = Y[Iu]
            # confidence
            Cu = 1.0 + self.alpha * np.ones(len(Iu))
            Cu_mius_I = Cu - 1.0

            # A = YtY + YI^T (Cu - I) YI + λI
            A = YTY + (YI.T * Cu_mius_I) @ YI + self.reg * np.eye(self.f)
            # b = Y^T C_u p_u = sum_i c_ui * y_i
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
            start, end = indptr[i], indptr[i+1]
            Ui = indices[start:end]
            XU = X[Ui]
            # confidence
            Cu = 1.0 + self.alpha * np.ones(len(Ui))
            Cu_mius_I = Cu - 1.0

            # A = XTX + XU^T (Cu - I) XU + λI
            A = XTX + (XU.T * Cu_mius_I) @ XU + self.reg * np.eye(self.f)
            # b = Y^T C_u p_u = sum_i c_ui * y_i
            b = (XU * Cu[:, None]).sum(axis=0)
            Y[i] = np.linalg.solve(A, b)
        self.item_factors = Y

    def fit(self, R: csr_matrix):
        n_users, n_items = R.shape
        self.user_factors = 0.01 * self.rng.standard_normal((n_users, self.f))
        self.item_factors = 0.01 * self.rng.standard_normal((n_items, self.f))

        for it in tqdm(range(1, self.n_iters + 1), desc="Progress"):
            self.als_user_step(R)
            self.als_item_step(R)
        
        return self
    
    def predict_user(self, user_id: int):
        return self.user_factors[user_id] @ self.item_factors.T

    def recommend(self, R: csr_matrix, user_id: int, user_idx_to_id: dict, item_idx_to_id: dict, k=20):
        scores = self.predict_user(user_id)
        start, end = R.indptr[user_id], R.indptr[user_id + 1]
        seen = set(R.indices[start: end])
        if seen:
            scores = scores.copy()
            scores[list(seen)] = -np.inf

        if k >= len(scores):
            topk = np.argsort(-scores)
        else:
            idx = np.argpartition(-scores, k)[:k]
            topk = idx[np.argsort(-scores[idx])]

        # map to original ids
        user_original_id = user_idx_to_id.get(user_id, user_id)
        item_original_ids = [item_idx_to_id.get(item_id, item_id) for item_id in topk[:k]]
        return user_original_id, item_original_ids, scores[topk]

if __name__ == "__main__":
    # get the data 
    df = load_dataset()
    csr, idx_to_user, idx_to_item = get_statistics_and_csr(df)
    train_csr, val_csr, test_csr = split_csr_train_val_test(csr, 0.1, 0.1)

    mf = ImplicitMF(n_factors=64, alpha=40.0, reg=0.1, n_iters = 32)
    mf.fit(train_csr)

    # Validate
    val_ndcg = ndcg_at_k_from_factors(R_test=val_csr, X=mf.user_factors, Y=mf.item_factors, R_train=train_csr, k=20)
    print(f"Val NDCG@20: {val_ndcg:.4f}")

    # Test
    test_ndcg = ndcg_at_k_from_factors(R_test = test_csr, X = mf.user_factors, Y = mf.item_factors, R_train=train_csr, k=20)
    print(f"Test NDCG@20: {test_ndcg:.4f}")

    u = 0
    user, items, scores = mf.recommend(train_csr, user_id=u, user_idx_to_id=idx_to_user, item_idx_to_id=idx_to_item, k=20)
    print("User", user, "top-10 items:", items)

    # predictions
    results = []
    for u in tqdm(range(csr.shape[0]), desc="Predictions Progress"):
        user, items, scores = mf.recommend(train_csr, user_id=u, user_idx_to_id=idx_to_user, item_idx_to_id=idx_to_item, k=20)
        results.append([user, *items])

    # write the results
    write_results_to_file(results, "output.txt")
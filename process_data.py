import numpy as np
from scipy.sparse import csr_matrix
from sklearn.utils import shuffle
import torch
import random
from torch.utils.data import Dataset


def load_dataset():
    """
    Read train-2.txt where each line is:
        user_id item_id1 item_id2 ...
    """
    user_interaction_data = {}
    with open("data/train-2.txt", "r") as file:
        for line in file:
            if line.strip() != "":
                row = [int(x) for x in line.strip().split()]
                user_interaction_data[row[0]] = row[1:]
    print("Loaded the dataset")
    print("First record:", user_interaction_data[0])
    return user_interaction_data


def get_statistics_and_csr(df: dict):
    print("\n\nStatistics")
    total_users = len(df)
    print(f"\n\nTotal users: {total_users}")

    # user lengths
    user_interaction_lengths = {key: len(value) for key, value in df.items()}
    sorted_user_interaction_lengths = sorted(
        user_interaction_lengths.items(), key=lambda x: x[1], reverse=True
    )
    print("Top 10 User Interaction Lengths:")
    for key, value in sorted_user_interaction_lengths[:5]:
        print(f"{key}: {value}")

    print("Bottom 10 User Interaction Lengths:")
    for key, value in sorted_user_interaction_lengths[-5:]:
        print(f"{key}: {value}")

    # item frequencies
    item_freqs = {}
    for items in df.values():
        for item in items:
            item_freqs[item] = item_freqs.get(item, 0) + 1
    total_items = len(item_freqs)
    print(f"\n\nTotal items: {total_items}")

    sorted_item_freqs = sorted(item_freqs.items(), key=lambda x: x[1], reverse=True)
    print("Top 10 Frequencies:")
    for key, value in sorted_item_freqs[:5]:
        print(f"{key}: {value}")

    print("Bottom 10 Frequencies:")
    for key, value in sorted_item_freqs[-5:]:
        print(f"{key}: {value}")

    # build csr matrix
    user_to_idx = {user: idx for idx, user in enumerate(df.keys())}
    item_to_idx = {item: idx for idx, item in enumerate(item_freqs.keys())}
    idx_to_user = {idx: user for user, idx in user_to_idx.items()}
    idx_to_item = {idx: item for item, idx in item_to_idx.items()}

    data, row_indices, col_indices = [], [], []
    for user, items in df.items():
        u_idx = user_to_idx[user]
        for item in items:
            i_idx = item_to_idx[item]
            row_indices.append(u_idx)
            col_indices.append(i_idx)
            data.append(1)

    csr = csr_matrix((data, (row_indices, col_indices)),
                     shape=(len(user_to_idx), len(item_to_idx)))

    print(f"\nCSR matrix shape: {csr.shape}")
    print(f"Non zero interactions: {csr.nnz}")

    return csr, idx_to_user, idx_to_item


def split_csr_train_val_test(
    csr: csr_matrix, val_ratio: float = 0.1, test_ratio: float = 0.1, seed: int = 42
):
    np.random.seed(seed)
    n_users, n_items = csr.shape

    train_rows, train_cols = [], []
    val_rows, val_cols = [], []
    test_rows, test_cols = [], []

    for user in range(n_users):
        start_ptr, end_ptr = csr.indptr[user], csr.indptr[user + 1]
        user_items = csr.indices[start_ptr:end_ptr]
        if len(user_items) == 0:
            continue

        user_items = shuffle(user_items, random_state=seed)
        n_total = len(user_items)
        n_val = int(n_total * val_ratio)
        n_test = int(n_total * test_ratio)

        val_items = user_items[:n_val]
        test_items = user_items[n_val : n_val + n_test]
        train_items = user_items[n_val + n_test :]

        train_rows.extend([user] * len(train_items))
        train_cols.extend(train_items)

        val_rows.extend([user] * len(val_items))
        val_cols.extend(val_items)

        test_rows.extend([user] * len(test_items))
        test_cols.extend(test_items)

    train_data = np.ones(len(train_rows))
    val_data = np.ones(len(val_rows))
    test_data = np.ones(len(test_rows))

    train_csr = csr_matrix((train_data, (train_rows, train_cols)), shape=(n_users, n_items))
    val_csr = csr_matrix((val_data, (val_rows, val_cols)), shape=(n_users, n_items))
    test_csr = csr_matrix((test_data, (test_rows, test_cols)), shape=(n_users, n_items))

    print(f"Train interactions: {train_csr.nnz}")
    print(f"Val interactions: {val_csr.nnz}")
    print(f"Test interactions: {test_csr.nnz}")

    return train_csr, val_csr, test_csr

def convert_csr_to_edge_list(csr_matrix):
    """Converts the training CSR matrix to a tensor of (user_id, item_id) pairs."""
    coo_matrix = csr_matrix.tocoo()
    users = torch.tensor(coo_matrix.row, dtype=torch.long)
    items = torch.tensor(coo_matrix.col, dtype=torch.long)
    return users, items

def build_index():
    """Index-based representation used by popularity + NeuralMF."""
    interactions = load_dataset()
    users = sorted(interactions.keys())
    items = sorted({i for S in interactions.values() for i in S})
    user2idx = {u: i for i, u in enumerate(users)}
    idx2user = {i: u for u, i in user2idx.items()}
    item2idx = {it: i for i, it in enumerate(items)}
    idx2item = {i: it for it, i in item2idx.items()}

    ui_pos = {
        user2idx[u]: set(item2idx[it] for it in items if it in item2idx)
        for u, items in interactions.items()
    }
    return ui_pos, user2idx, idx2user, item2idx, idx2item


def train_valid_split(ui_pos, valid_ratio=0.1, seed=42):
    rng = random.Random(seed)
    train = {}
    valid = {}
    for u, items in ui_pos.items():
        if len(items) == 1:
            train[u] = set(items)
            valid[u] = set()
            continue
        n_valid = max(1, int(len(items) * valid_ratio))
        items = list(items)
        rng.shuffle(items)
        valid_items = set(items[:n_valid])
        train_items = set(items[n_valid:])
        train[u] = train_items
        valid[u] = valid_items
    return train, valid


class PairDataset(Dataset):
    """
    Used for NeuralMF:
    returns (user, positive_item, [negatives...])
    """

    def __init__(self, ui_pos, num_items, num_neg=4, pop_weights=None):
        self.ui_pos = [(u, i) for u, items in ui_pos.items() for i in items]
        self.user_pos = {u: set(items) for u, items in ui_pos.items()}
        self.num_items = num_items
        self.num_neg = num_neg
        self.pop_weights = pop_weights  # numpy array shape [num_items], or None

    def __len__(self):
        return len(self.ui_pos)

    def sample_neg(self, u):
        while True:
            if self.pop_weights is not None:
                j = np.random.choice(self.num_items, p=self.pop_weights)
            else:
                j = random.randint(0, self.num_items - 1)
            if j not in self.user_pos[u]:
                return j

    def __getitem__(self, idx):
        u, i = self.ui_pos[idx]
        negs = [self.sample_neg(u) for _ in range(self.num_neg)]
        return u, i, torch.tensor(negs, dtype=torch.long)


if __name__ == "__main__":
    df = load_dataset()
    csr, _, _ = get_statistics_and_csr(df)
    train_csr, val_csr, test_csr = split_csr_train_val_test(csr, 0.1, 0.1)

import os
import numpy as np
import scipy.sparse as sp
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from sklearn.decomposition import TruncatedSVD

BASE_PATH = "./data/corpora/eng_wikipedia_2016_1M"
WORDS_PATH = os.path.join(BASE_PATH, "eng_wikipedia_2016_1M-words.txt")
WORDSIM_PATH = "./data/wordsim353.csv"

INPUT_MATRICES_DIR = "./data/results3/matrices"   # existing PPMI
OUTPUT_DIR = "./data/results4"                    # NEW: pruned results
PLOTS_DIR = os.path.join(OUTPUT_DIR, "plots")
MATRICES_DIR = os.path.join(OUTPUT_DIR, "matrices")
os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(MATRICES_DIR, exist_ok=True)


def load_vocab_full(path, top_k=30000):
    raw_counts = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 2:
                word = parts[1]
                count = int(parts[2]) if len(parts) >= 3 else 1
                raw_counts.append((word, count))
    raw_counts.sort(key=lambda x: x[1], reverse=True)
    truncated = raw_counts[:top_k]
    vocab = {}
    inverse_vocab = {}
    for idx, (w, _) in enumerate(truncated):
        vocab[w] = idx
        inverse_vocab[idx] = w
    print(f"[Info] Vocab size={len(vocab)}")
    return vocab, inverse_vocab


class MatrixTransformer:
    @staticmethod
    def load_ppmi(window_size):
        p1 = os.path.join(INPUT_MATRICES_DIR, f"ppmi_w{window_size}.npz")
        if os.path.exists(p1):
            print(f"[Cache] Loading PPMI (W={window_size}) from results1...")
            return sp.load_npz(p1)
        raise FileNotFoundError(p1)

    @staticmethod
    def prune_and_svd(ppmi_mat, inverse_vocab, window_size, fixed_d=300):
        N = ppmi_mat.shape[0]
        row_sums = np.array(ppmi_mat.sum(axis=1)).ravel()
        alive_mask = row_sums > 0
        alive_indices = np.where(alive_mask)[0]
        num_alive = len(alive_indices)
        num_dead = N - num_alive
        print(f"[Prune] {num_alive} alive, {num_dead} dead rows (row_sum==0)")

        # pruned PPMI (only alive rows)
        ppmi_pruned = ppmi_mat[alive_indices]

        # SVD on pruned matrix
        k = min(fixed_d, ppmi_pruned.shape[1] - 1)
        print(f"[SVD] TruncatedSVD on shape {ppmi_pruned.shape} with d={k}")
        svd = TruncatedSVD(n_components=k, n_iter=5, random_state=42)
        emb_pruned = svd.fit_transform(ppmi_pruned)
        s_vals = svd.singular_values_

        # build pruned vocab list in correct order
        vocab_pruned = [inverse_vocab[i] for i in alive_indices]

        # save
        np.save(os.path.join(MATRICES_DIR, f"svd_pruned_w{window_size}.npy"), emb_pruned)
        np.save(os.path.join(MATRICES_DIR, f"alive_indices_w{window_size}.npy"), alive_indices)
        np.save(os.path.join(MATRICES_DIR, f"singular_values_pruned_w{window_size}.npy"), s_vals)
        with open(os.path.join(MATRICES_DIR, f"vocab_pruned_w{window_size}.txt"),
                  "w", encoding="utf-8") as f:
            for w in vocab_pruned:
                f.write(w + "\n")

        print(f"[Save] SVD & masks saved for W={window_size}")
        return emb_pruned, ppmi_pruned, vocab_pruned


class MetricEvaluator:
    def __init__(self, vocab):
        self.vocab = vocab

    def _cosine_sim_batch(self, matrix, pairs):
        is_sparse = sp.issparse(matrix)
        if is_sparse:
            norms = sp.linalg.norm(matrix, axis=1)
            norms[norms == 0] = 1
            norm_matrix = sp.diags(1.0 / norms) @ matrix
        else:
            norms = np.linalg.norm(matrix, axis=1, keepdims=True)
            norms[norms == 0] = 1
            norm_matrix = matrix / norms

        valid_indices, human_scores = [], []
        for w1, w2, score in pairs:
            if w1 in self.vocab and w2 in self.vocab:
                valid_indices.append((self.vocab[w1], self.vocab[w2]))
                human_scores.append(score)
        if not valid_indices:
            return 0.0, 0

        model_scores = []
        for i1, i2 in valid_indices:
            if is_sparse:
                sim = (norm_matrix[i1] @ norm_matrix[i2].T)[0, 0]
            else:
                sim = float(np.dot(norm_matrix[i1], norm_matrix[i2]))
            model_scores.append(sim)
        return spearmanr(human_scores, model_scores)[0], len(human_scores)

    def evaluate_wordsim(self, matrix, path):
        if not os.path.exists(path):
            return 0.0, 0
        df = pd.read_csv(path)
        cols = [c.lower() for c in df.columns]
        w1_idx = next(i for i, c in enumerate(cols) if "word 1" in c or "word1" in c)
        w2_idx = next(i for i, c in enumerate(cols) if "word 2" in c or "word2" in c)
        s_idx = next(i for i, c in enumerate(cols) if "human" in c or "score" in c)
        pairs = [(str(r[w1_idx]), str(r[w2_idx]), float(r[s_idx]))
                 for r in df.itertuples(index=False)]
        return self._cosine_sim_batch(matrix, pairs)


def run_all():
    vocab, inverse_vocab = load_vocab_full(WORDS_PATH, top_k=30000)
    evaluator = MetricEvaluator(vocab)   # for PPMI metrics on full vocab

    windows = [7]  # or [5,6,7,8,9,10]
    for w in windows:
        print(f"\n=== Window {w} ===")
        ppmi_mat = MatrixTransformer.load_ppmi(w)

        # metrics on full PPMI (optional)
        corr_ppmi, _ = evaluator.evaluate_wordsim(ppmi_mat, WORDSIM_PATH)
        print(f"[Metric] PPMI WordSim corr={corr_ppmi:.4f}")

        # pruned SVD
        emb_pruned, ppmi_pruned, vocab_pruned = MatrixTransformer.prune_and_svd(
            ppmi_mat, inverse_vocab, w, fixed_d=300
        )

        # build vocab->index mapping in pruned space for SVD metrics
        pruned_vocab_map = {wrd: i for i, wrd in enumerate(vocab_pruned)}
        evaluator_pruned = MetricEvaluator(pruned_vocab_map)
        corr_svd, _ = evaluator_pruned.evaluate_wordsim(emb_pruned, WORDSIM_PATH)
        print(f"[Metric] SVD(pruned) WordSim corr={corr_svd:.4f}")


if __name__ == "__main__":
    run_all()

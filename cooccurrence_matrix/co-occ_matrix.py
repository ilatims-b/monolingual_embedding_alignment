import os
import re
import numpy as np
import scipy.sparse as sp
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
from scipy.stats import spearmanr
from sklearn.decomposition import TruncatedSVD

BASE_PATH = "./data/corpora/eng_wikipedia_2016_1M"
SENTENCES_PATH = os.path.join(BASE_PATH, "eng_wikipedia_2016_1M-sentences.txt")
WORDS_PATH = os.path.join(BASE_PATH, "eng_wikipedia_2016_1M-words.txt")
WORDSIM_PATH = "./data/wordsim353.csv"
OUTPUT_DIR = "./data/results3"
PLOTS_DIR = os.path.join(OUTPUT_DIR, "plots")
MATRICES_DIR = os.path.join(OUTPUT_DIR, "matrices")

os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(MATRICES_DIR, exist_ok=True)

class MatrixBuilder:
    def __init__(self):
        self.vocab = {}
        self.inverse_vocab = {}
        
    def load_vocab(self, path, top_k=25000):
        print(f"[Info] Loading vocabulary from {path} (Top-{top_k})...")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Vocab file not found: {path}")
            
        #(No merging, no lowercasing, preserve case & numbers)
        raw_counts = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 2:
                    word = parts[1]
                    count = int(parts[2]) if len(parts) >= 3 else 1
                    raw_counts.append((word, count))

        # Sort by Frequency Descending
        raw_counts.sort(key=lambda x: x[1], reverse=True)
        
        # Truncate
        truncated = raw_counts[:top_k]
        
        # Build Index Maps
        self.vocab = {}
        self.inverse_vocab = {}
        for idx, (word, _) in enumerate(truncated):
            self.vocab[word] = idx
            self.inverse_vocab[idx] = word
            
        print(f"[Info] Vocabulary loaded. Size: {len(self.vocab)}")
        if truncated:
            print(f"[Info] Cutoff frequency at rank {top_k}: {truncated[-1][1]}")

    def preprocess(self, text):
        # split by whitespace, then strip ONLY leading/trailing punctuation that are NOT part of the word.
        # preserves "1.8", "co-operate", "Anne's" but removes "word," or "(word)".
        
        tokens = text.split()
        clean_tokens = []
        for t in tokens:
            # Strip common sentence punctuation from edges
            t = t.strip('.,?"!;:()[]{}') 
            if t:
                clean_tokens.append(t)
        return clean_tokens

    def build_cooccurrence(self, sentences, window_size):
        if not self.vocab: raise ValueError("Vocab empty")
        
        save_path = os.path.join(MATRICES_DIR, f"cooc_w{window_size}.npz")
        if os.path.exists(save_path):
            print(f"[Cache] Loading Co-occurrence Matrix (W={window_size}) from disk...")
            return sp.load_npz(save_path)

        if not sentences:
             raise ValueError(f"Matrix W={window_size} missing but corpus is empty! Logic Error.")

        print(f"[Process] Building Co-occurrence Matrix (W={window_size}) from scratch...")
        cooccurrences = Counter()
        N = len(self.vocab)
        
        for idx, sent in enumerate(sentences):
            parts = sent.strip().split('\t')
            text = parts[1] if len(parts) > 1 else parts[0]
            
            # Preprocess text
            tokens = [t for t in self.preprocess(text) if t in self.vocab]
            
            for i, target in enumerate(tokens):
                t_idx = self.vocab[target]
                start = max(0, i - window_size)
                end = min(len(tokens), i + window_size + 1)
                for j in range(start, end):
                    if i == j: continue
                    c_idx = self.vocab[tokens[j]]
                    cooccurrences[(t_idx, c_idx)] += 1
            
            if idx % 100000 == 0 and idx > 0:
                print(f"  Processed {idx} sentences...")

        data = list(cooccurrences.values())
        indices = list(cooccurrences.keys())
        rows, cols = zip(*indices) if indices else ([], [])
        
        mat = sp.csr_matrix((data, (rows, cols)), shape=(N, N), dtype=np.float32)
        sp.save_npz(save_path, mat)
        return mat

class MatrixTransformer:
    @staticmethod
    def to_ppmi(co_matrix, window_size):
        save_path = os.path.join(MATRICES_DIR, f"ppmi_w{window_size}.npz")
        if os.path.exists(save_path):
            print(f"[Cache] Loading PPMI Matrix (W={window_size})...")
            return sp.load_npz(save_path)
            
        print("[Process] Calculating PPMI...")
        total = co_matrix.sum()
        row_sums = np.array(co_matrix.sum(axis=1)).flatten()
        col_sums = np.array(co_matrix.sum(axis=0)).flatten()
        
        row_sums[row_sums == 0] = 1
        col_sums[col_sums == 0] = 1
        
        ppmi = co_matrix.copy().tocoo()
        row_probs = row_sums[ppmi.row]
        col_probs = col_sums[ppmi.col]
        
        pmi_vals = np.log2((ppmi.data * total) / (row_probs * col_probs + 1e-8))
        mask = pmi_vals > 0
        
        mat = sp.csr_matrix((pmi_vals[mask], (ppmi.row[mask], ppmi.col[mask])), shape=co_matrix.shape)
        sp.save_npz(save_path, mat)
        return mat

    # @staticmethod
    # def determine_optimal_d_and_embed(matrix, window_size, max_scan=300):
    #     save_path = os.path.join(MATRICES_DIR, f"svd_w{window_size}.npy")
    #     singular_vals_path = os.path.join(MATRICES_DIR, f"singular_values_w{window_size}.npy")
        
    #     if os.path.exists(save_path) and os.path.exists(singular_vals_path):
    #         print(f"[Cache] Loading SVD Embeddings (W={window_size})...")
    #         return np.load(save_path), np.load(singular_vals_path)
            
    #     print(f"[Process] Calculating SVD Spectrum (scan_dim={max_scan})...")
    #     k = min(max_scan, matrix.shape[1] - 1)
    #     svd = TruncatedSVD(n_components=k, n_iter=5, random_state=42)
    #     svd.fit(matrix)
        
    #     s_vals = svd.singular_values_
    #     # total_var = np.sum(svd.explained_variance_)
    #     # cum_var = np.cumsum(svd.explained_variance_)
        
    #     # optimal_k = np.searchsorted(cum_var, 0.80 * total_var) + 1
    #     # optimal_k = max(optimal_k, 50) 
    #     optimal_k=300
    #     print(f"  [Auto-SVD] Selected d={optimal_k} (explains 80% of top-{k} variance)")
        
    #     embeddings = svd.fit_transform(matrix)[:, :optimal_k]
        
    #     np.save(save_path, embeddings)
    #     np.save(singular_vals_path, s_vals)
    #     return embeddings, s_vals
    @staticmethod
    def determine_optimal_d_and_embed(matrix, window_size, fixed_d=300):
        save_path = os.path.join(MATRICES_DIR, f"svd_w{window_size}.npy")
        singular_vals_path = os.path.join(MATRICES_DIR, f"singular_values_w{window_size}.npy")
        
        if os.path.exists(save_path) and os.path.exists(singular_vals_path):
            print(f"[Cache] Loading SVD Embeddings (W={window_size})...")
            return np.load(save_path), np.load(singular_vals_path)
            
        k = min(fixed_d, matrix.shape[1] - 1)
        
        print(f"[Process] Calculating SVD (fixed d={k})...")
        svd = TruncatedSVD(n_components=k, n_iter=5, random_state=42)
        
        embeddings = svd.fit_transform(matrix)
        
        s_vals = svd.singular_values_
        
        print(f"  [SVD] Fixed dimension d={k} used.")
        
        np.save(save_path, embeddings)
        np.save(singular_vals_path, s_vals)
        return embeddings, s_vals


class MetricEvaluator:
    def __init__(self, vocab):
        self.vocab = vocab

    def _cosine_sim_batch(self, matrix, pairs):
        is_sparse = sp.issparse(matrix)
        scores = []
        if is_sparse:
            norms = sp.linalg.norm(matrix, axis=1)
            norms[norms==0] = 1
            norm_matrix = sp.diags(1.0/norms) @ matrix
        else:
            norms = np.linalg.norm(matrix, axis=1, keepdims=True)
            norms[norms==0] = 1
            norm_matrix = matrix / norms

        valid_indices = []
        human_scores = []
        for w1, w2, score in pairs:
            if w1 in self.vocab and w2 in self.vocab:
                valid_indices.append((self.vocab[w1], self.vocab[w2]))
                human_scores.append(score)
        
        if not valid_indices: return 0.0, 0
        model_scores = []
        for idx1, idx2 in valid_indices:
            if is_sparse: sim = (norm_matrix[idx1] @ norm_matrix[idx2].T)[0,0]
            else: sim = np.dot(norm_matrix[idx1], norm_matrix[idx2])
            model_scores.append(sim)
        return spearmanr(human_scores, model_scores)[0], len(human_scores)

    def evaluate_wordsim(self, matrix, path):
        if not os.path.exists(path): return 0.0
        df = pd.read_csv(path)
        cols = [c.lower() for c in df.columns]
        w1_idx = next(i for i, c in enumerate(cols) if 'word 1' in c or 'word1' in c)
        w2_idx = next(i for i, c in enumerate(cols) if 'word 2' in c or 'word2' in c)
        s_idx = next(i for i, c in enumerate(cols) if 'human' in c or 'score' in c)
        pairs = []
        for row in df.itertuples(index=False):
            pairs.append((str(row[w1_idx]), str(row[w2_idx]), float(row[s_idx])))
        return self._cosine_sim_batch(matrix, pairs)

    def get_neighbors(self, matrix, word_list, k=10):
        neighbors = {}
        is_sparse = sp.issparse(matrix)
        if is_sparse:
            norms = sp.linalg.norm(matrix, axis=1)
            norms[norms==0] = 1
            norm_matrix = sp.diags(1.0/norms) @ matrix
        else:
            norms = np.linalg.norm(matrix, axis=1, keepdims=True)
            norms[norms==0] = 1
            norm_matrix = matrix / norms
            
        for word in word_list:
            if word not in self.vocab: continue
            idx = self.vocab[word]
            if is_sparse: scores = (norm_matrix[idx] @ norm_matrix.T).toarray().flatten()
            else: scores = np.dot(norm_matrix[idx], norm_matrix.T)
            top_k = np.argsort(scores)[-(k+1):][::-1]
            neighbors[word] = set(i for i in top_k if i != idx)
        return neighbors

    def calculate_drift(self, base_neighbors, curr_neighbors):
        drifts = []
        for word, base_set in base_neighbors.items():
            if word in curr_neighbors:
                curr_set = curr_neighbors[word]
                union = len(base_set.union(curr_set))
                if union > 0:
                    jaccard = len(base_set.intersection(curr_set)) / union
                    drifts.append(1.0 - jaccard)
                else: drifts.append(1.0)
        return np.mean(drifts) if drifts else 0.0

    def calculate_sparsity(self, matrix):
        if sp.issparse(matrix):
            return matrix.nnz / (matrix.shape[0] * matrix.shape[1])
        return 1.0

def save_plots(results_df, output_dir, singular_vals_dict):
    windows = results_df['window']
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    ax1.plot(windows, results_df['corr_ppmi'], 'b-o', label='Semantic')
    ax1.plot(windows, 1.0 - results_df['drift_ppmi'], 'r--s', label='Stability')
    ax1.plot(windows, results_df['sci_ppmi'], 'g-^', linewidth=2, label='SCI')
    ax1.set_title('PPMI: Differential SCI')
    ax1.legend(); ax1.grid(True)
    ax2.plot(windows, results_df['corr_svd'], 'c-o', label='Semantic')
    ax2.plot(windows, 1.0 - results_df['drift_svd'], 'm--s', label='Stability')
    ax2.plot(windows, results_df['sci_svd'], 'k-^', linewidth=2, label='SCI')
    ax2.set_title('SVD: Differential SCI')
    ax2.legend(); ax2.grid(True)
    plt.tight_layout(); plt.savefig(os.path.join(output_dir, "sci_breakdown_differential.png")); plt.close()

def main():
    print("=== Starting Co-occurrence Matrix Experiment (Differential Drift) ===")
    
    builder = MatrixBuilder()
    builder.load_vocab(WORDS_PATH, top_k=30000)
    
    windows = [5,6,7,8,9,10]
    
    missing_cache = False
    for w in windows:
        if not os.path.exists(os.path.join(MATRICES_DIR, f"cooc_w{w}.npz")):
            missing_cache = True
            break
            
    if missing_cache:
        print("[Info] Some matrices missing from cache. Loading corpus...")
        with open(SENTENCES_PATH, "r", encoding="utf-8") as f:
            corpus = f.readlines()
    else:
        print("[Info] All matrices found in cache. Corpus load skipped.")
        corpus = []
        
    evaluator = MetricEvaluator(builder.vocab)
    test_words = []
    for i in range(len(builder.inverse_vocab)):
        w = builder.inverse_vocab[i]
        test_words.append(w)
        if len(test_words) >= 100:
            break
            
    results = []
    singular_vals_history = {}
    base_neighbors_ppmi = None
    base_neighbors_svd = None

    for w in windows:
        print(f"\n--- Processing Window: {w} ---")
        co_mat = builder.build_cooccurrence(corpus, w)
        ppmi_mat = MatrixTransformer.to_ppmi(co_mat, w)
        svd_mat, s_vals = MatrixTransformer.determine_optimal_d_and_embed(ppmi_mat, w)
        singular_vals_history[w] = s_vals
        
        corr_ppmi, _ = evaluator.evaluate_wordsim(ppmi_mat, WORDSIM_PATH)
        curr_neighbors_ppmi = evaluator.get_neighbors(ppmi_mat, test_words)
        drift_ppmi = evaluator.calculate_drift(base_neighbors_ppmi, curr_neighbors_ppmi) if base_neighbors_ppmi else 0.0
        base_neighbors_ppmi = curr_neighbors_ppmi
        
        sem = max(0, corr_ppmi)
        syn = 1.0 - drift_ppmi
        sci_ppmi = 2 * (sem * syn) / (sem + syn) if (sem + syn) > 0 else 0
        
        corr_svd, _ = evaluator.evaluate_wordsim(svd_mat, WORDSIM_PATH)
        curr_neighbors_svd = evaluator.get_neighbors(svd_mat, test_words)
        drift_svd = evaluator.calculate_drift(base_neighbors_svd, curr_neighbors_svd) if base_neighbors_svd else 0.0
        base_neighbors_svd = curr_neighbors_svd
            
        sem_svd = max(0, corr_svd)
        syn_svd = 1.0 - drift_svd
        sci_svd = 2 * (sem_svd * syn_svd) / (sem_svd + syn_svd) if (sem_svd + syn_svd) > 0 else 0
        
        sparsity = evaluator.calculate_sparsity(ppmi_mat)
        optimal_d = svd_mat.shape[1]
        
        row = {"window": w, "optimal_d": optimal_d, "sparsity_ppmi": sparsity, "corr_ppmi": corr_ppmi, "drift_ppmi": drift_ppmi, "sci_ppmi": sci_ppmi, "corr_svd": corr_svd, "drift_svd": drift_svd, "sci_svd": sci_svd}
        results.append(row)
        print(f"  [Result] PPMI SCI: {sci_ppmi:.4f} | SVD SCI: {sci_svd:.4f} | d={optimal_d}")

    df = pd.DataFrame(results)
    csv_path = os.path.join(OUTPUT_DIR, "metrics_summary.csv")
    df.to_csv(csv_path, index=False)
    print(f"\n[Info] Metrics saved to {csv_path}")
    save_plots(df, PLOTS_DIR, singular_vals_history)
    print(f"[Info] Plots saved to {PLOTS_DIR}")

if __name__ == "__main__":
    main()


#results3 - svd d=300 (n=5,6,7,8,9,10)
#results1- svd d= 80% of variance in d=300 (n=1,2,5,7,10,12,15,17,20)
#results 4 - pruned svd 7
#results 2 - svd d=300 (n=1,2,5,7,10,12,15,17,20)

# import os
# import re
# import numpy as np
# import scipy.sparse as sp
# import pandas as pd
# import matplotlib.pyplot as plt
# from collections import Counter
# from scipy.stats import spearmanr
# from sklearn.decomposition import TruncatedSVD


# BASE_PATH = "./data/corpora/eng_wikipedia_2016_1M"
# SENTENCES_PATH = os.path.join(BASE_PATH, "eng_wikipedia_2016_1M-sentences.txt")
# WORDS_PATH = os.path.join(BASE_PATH, "eng_wikipedia_2016_1M-words.txt")
# WORDSIM_PATH = "./data/wordsim353.csv"

# # INPUT DIR (Where PPMI matrices are already stored)
# INPUT_MATRICES_DIR = "./data/results1/matrices"

# # OUTPUT DIR (Where new SVD results will go)
# OUTPUT_DIR = "./data/results2"
# PLOTS_DIR = os.path.join(OUTPUT_DIR, "plots")
# MATRICES_DIR = os.path.join(OUTPUT_DIR, "matrices") # New SVDs go here

# os.makedirs(PLOTS_DIR, exist_ok=True)
# os.makedirs(MATRICES_DIR, exist_ok=True)


# class MatrixBuilder:
#     def __init__(self):
#         self.vocab = {}
#         self.inverse_vocab = {}
        
#     def load_vocab(self, path, top_k=25000):
#         print(f"[Info] Loading vocabulary from {path} (Top-{top_k})...")
#         if not os.path.exists(path):
#             raise FileNotFoundError(f"Vocab file not found: {path}")
            
#         raw_counts = []
#         with open(path, "r", encoding="utf-8") as f:
#             for line in f:
#                 parts = line.strip().split('\t')
#                 if len(parts) >= 2:
#                     word = parts[1]
#                     count = int(parts[2]) if len(parts) >= 3 else 1
#                     raw_counts.append((word, count))

#         raw_counts.sort(key=lambda x: x[1], reverse=True)
#         truncated = raw_counts[:top_k]
        
#         self.vocab = {}
#         self.inverse_vocab = {}
#         for idx, (word, _) in enumerate(truncated):
#             self.vocab[word] = idx
#             self.inverse_vocab[idx] = word
            
#         print(f"[Info] Vocabulary loaded. Size: {len(self.vocab)}")
#         if truncated:
#             print(f"[Info] Cutoff frequency at rank {top_k}: {truncated[-1][1]}")

#     def preprocess(self, text):
#         tokens = text.split()
#         clean_tokens = []
#         for t in tokens:
#             t = t.strip('.,?"!;:()[]{}') 
#             if t:
#                 clean_tokens.append(t)
#         return clean_tokens

#     def build_cooccurrence(self, sentences, window_size):
#         # We try to load from INPUT_MATRICES_DIR (results1) first
#         input_path = os.path.join(INPUT_MATRICES_DIR, f"cooc_w{window_size}.npz")
#         if os.path.exists(input_path):
#              print(f"[Cache] Loading Co-occurrence Matrix (W={window_size}) from results1...")
#              return sp.load_npz(input_path)

#         save_path = os.path.join(MATRICES_DIR, f"cooc_w{window_size}.npz")
#         if os.path.exists(save_path):
#              return sp.load_npz(save_path)

#         if not sentences:
#              raise ValueError(f"Matrix W={window_size} missing in {INPUT_MATRICES_DIR}!")

#         print(f"[Process] Building Co-occurrence Matrix (W={window_size}) from scratch...")
#         cooccurrences = Counter()
#         N = len(self.vocab)
        
#         for idx, sent in enumerate(sentences):
#             parts = sent.strip().split('\t')
#             text = parts[1] if len(parts) > 1 else parts[0]
#             tokens = [t for t in self.preprocess(text) if t in self.vocab]
#             for i, target in enumerate(tokens):
#                 t_idx = self.vocab[target]
#                 start = max(0, i - window_size)
#                 end = min(len(tokens), i + window_size + 1)
#                 for j in range(start, end):
#                     if i == j: continue
#                     c_idx = self.vocab[tokens[j]]
#                     cooccurrences[(t_idx, c_idx)] += 1
                    
#         data = list(cooccurrences.values())
#         indices = list(cooccurrences.keys())
#         rows, cols = zip(*indices) if indices else ([], [])
#         mat = sp.csr_matrix((data, (rows, cols)), shape=(N, N), dtype=np.float32)
#         # Save to NEW dir if we had to rebuild
#         sp.save_npz(save_path, mat)
#         return mat


# class MatrixTransformer:
#     @staticmethod
#     def to_ppmi(co_matrix, window_size):

#         input_path = os.path.join(INPUT_MATRICES_DIR, f"ppmi_w{window_size}.npz")
#         if os.path.exists(input_path):
#             print(f"[Cache] Loading PPMI Matrix (W={window_size}) from results1...")
#             return sp.load_npz(input_path)
            

#         save_path = os.path.join(MATRICES_DIR, f"ppmi_w{window_size}.npz")
#         if os.path.exists(save_path):
#              return sp.load_npz(save_path)


#         print("[Process] Calculating PPMI...")
#         total = co_matrix.sum()
#         row_sums = np.array(co_matrix.sum(axis=1)).flatten()
#         col_sums = np.array(co_matrix.sum(axis=0)).flatten()
#         row_sums[row_sums == 0] = 1
#         col_sums[col_sums == 0] = 1
#         ppmi = co_matrix.copy().tocoo()
#         row_probs = row_sums[ppmi.row]
#         col_probs = col_sums[ppmi.col]
#         pmi_vals = np.log2((ppmi.data * total) / (row_probs * col_probs + 1e-8))
#         mask = pmi_vals > 0
#         mat = sp.csr_matrix((pmi_vals[mask], (ppmi.row[mask], ppmi.col[mask])), shape=co_matrix.shape)
        
#         sp.save_npz(save_path, mat) # Save to results2
#         return mat

#     @staticmethod
#     def determine_optimal_d_and_embed(matrix, window_size, fixed_d=300):

#         save_path = os.path.join(MATRICES_DIR, f"svd_w{window_size}.npy")
#         singular_vals_path = os.path.join(MATRICES_DIR, f"singular_values_w{window_size}.npy")
        
#         if os.path.exists(save_path) and os.path.exists(singular_vals_path):
#             print(f"[Cache] Loading SVD Embeddings (W={window_size}) from results2...")
#             return np.load(save_path), np.load(singular_vals_path)
            
#         k = min(fixed_d, matrix.shape[1] - 1)
#         print(f"[Process] Calculating SVD (fixed d={k})...")
        
#         svd = TruncatedSVD(n_components=k, n_iter=5, random_state=42)
#         embeddings = svd.fit_transform(matrix) # Single pass
#         s_vals = svd.singular_values_
        
#         print(f"  [SVD] Fixed dimension d={k} used.")
        
#         np.save(save_path, embeddings)
#         np.save(singular_vals_path, s_vals)
#         return embeddings, s_vals


# class MetricEvaluator:
#     def __init__(self, vocab):
#         self.vocab = vocab

#     def _cosine_sim_batch(self, matrix, pairs):
#         is_sparse = sp.issparse(matrix)
#         scores = []
#         if is_sparse:
#             norms = sp.linalg.norm(matrix, axis=1)
#             norms[norms==0] = 1
#             norm_matrix = sp.diags(1.0/norms) @ matrix
#         else:
#             norms = np.linalg.norm(matrix, axis=1, keepdims=True)
#             norms[norms==0] = 1
#             norm_matrix = matrix / norms

#         valid_indices = []
#         human_scores = []
#         for w1, w2, score in pairs:
#             if w1 in self.vocab and w2 in self.vocab:
#                 valid_indices.append((self.vocab[w1], self.vocab[w2]))
#                 human_scores.append(score)
        
#         if not valid_indices: return 0.0, 0
#         model_scores = []
#         for idx1, idx2 in valid_indices:
#             if is_sparse: sim = (norm_matrix[idx1] @ norm_matrix[idx2].T)[0,0]
#             else: sim = np.dot(norm_matrix[idx1], norm_matrix[idx2])
#             model_scores.append(sim)
#         return spearmanr(human_scores, model_scores)[0], len(human_scores)

#     def evaluate_wordsim(self, matrix, path):
#         if not os.path.exists(path): return 0.0
#         df = pd.read_csv(path)
#         cols = [c.lower() for c in df.columns]
#         w1_idx = next(i for i, c in enumerate(cols) if 'word 1' in c or 'word1' in c)
#         w2_idx = next(i for i, c in enumerate(cols) if 'word 2' in c or 'word2' in c)
#         s_idx = next(i for i, c in enumerate(cols) if 'human' in c or 'score' in c)
#         pairs = []
#         for row in df.itertuples(index=False):
#             pairs.append((str(row[w1_idx]), str(row[w2_idx]), float(row[s_idx])))
#         return self._cosine_sim_batch(matrix, pairs)

#     def get_neighbors(self, matrix, word_list, k=10):
#         neighbors = {}
#         is_sparse = sp.issparse(matrix)
#         if is_sparse:
#             norms = sp.linalg.norm(matrix, axis=1)
#             norms[norms==0] = 1
#             norm_matrix = sp.diags(1.0/norms) @ matrix
#         else:
#             norms = np.linalg.norm(matrix, axis=1, keepdims=True)
#             norms[norms==0] = 1
#             norm_matrix = matrix / norms
            
#         for word in word_list:
#             if word not in self.vocab: continue
#             idx = self.vocab[word]
#             if is_sparse: scores = (norm_matrix[idx] @ norm_matrix.T).toarray().flatten()
#             else: scores = np.dot(norm_matrix[idx], norm_matrix.T)
#             top_k = np.argsort(scores)[-(k+1):][::-1]
#             neighbors[word] = set(i for i in top_k if i != idx)
#         return neighbors

#     def calculate_drift(self, base_neighbors, curr_neighbors):
#         drifts = []
#         for word, base_set in base_neighbors.items():
#             if word in curr_neighbors:
#                 curr_set = curr_neighbors[word]
#                 union = len(base_set.union(curr_set))
#                 if union > 0:
#                     jaccard = len(base_set.intersection(curr_set)) / union
#                     drifts.append(1.0 - jaccard)
#                 else: drifts.append(1.0)
#         return np.mean(drifts) if drifts else 0.0

#     def calculate_sparsity(self, matrix):
#         if sp.issparse(matrix):
#             return matrix.nnz / (matrix.shape[0] * matrix.shape[1])
#         return 1.0


# def save_plots(results_df, output_dir, singular_vals_dict):
#     windows = results_df['window']
#     fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
#     ax1.plot(windows, results_df['corr_ppmi'], 'b-o', label='Semantic')
#     ax1.plot(windows, 1.0 - results_df['drift_ppmi'], 'r--s', label='Stability')
#     ax1.plot(windows, results_df['sci_ppmi'], 'g-^', linewidth=2, label='SCI')
#     ax1.set_title('PPMI: Differential SCI')
#     ax1.legend(); ax1.grid(True)
#     ax2.plot(windows, results_df['corr_svd'], 'c-o', label='Semantic')
#     ax2.plot(windows, 1.0 - results_df['drift_svd'], 'm--s', label='Stability')
#     ax2.plot(windows, results_df['sci_svd'], 'k-^', linewidth=2, label='SCI')
#     ax2.set_title('SVD: Differential SCI')
#     ax2.legend(); ax2.grid(True)
#     plt.tight_layout(); plt.savefig(os.path.join(output_dir, "sci_breakdown_differential.png")); plt.close()


# def main():
#     print("=== Starting SVD Re-Calculation (Fixed d=300) ===")
    
#     builder = MatrixBuilder()
#     builder.load_vocab(WORDS_PATH, top_k=25000)
    
#     corpus = [] 
    
#     evaluator = MetricEvaluator(builder.vocab)
#     test_words = []
#     for i in range(len(builder.inverse_vocab)):
#         w = builder.inverse_vocab[i]
#         test_words.append(w)
#         if len(test_words) >= 100:
#             break
            
#     results = []
#     singular_vals_history = {}
#     base_neighbors_ppmi = None
#     base_neighbors_svd = None
    
#     windows = [1, 2, 5, 7, 10, 12, 15, 17, 20]

#     for w in windows:
#         print(f"\n--- Processing Window: {w} ---")
        

#         co_mat = builder.build_cooccurrence(corpus, w)
#         ppmi_mat = MatrixTransformer.to_ppmi(co_mat, w)
        
#         svd_mat, s_vals = MatrixTransformer.determine_optimal_d_and_embed(ppmi_mat, w, fixed_d=300)
#         singular_vals_history[w] = s_vals
        

#         corr_ppmi, _ = evaluator.evaluate_wordsim(ppmi_mat, WORDSIM_PATH)
#         curr_neighbors_ppmi = evaluator.get_neighbors(ppmi_mat, test_words)
#         drift_ppmi = evaluator.calculate_drift(base_neighbors_ppmi, curr_neighbors_ppmi) if base_neighbors_ppmi else 0.0
#         base_neighbors_ppmi = curr_neighbors_ppmi
        
#         sem = max(0, corr_ppmi)
#         syn = 1.0 - drift_ppmi
#         sci_ppmi = 2 * (sem * syn) / (sem + syn) if (sem + syn) > 0 else 0
        
#         corr_svd, _ = evaluator.evaluate_wordsim(svd_mat, WORDSIM_PATH)
#         curr_neighbors_svd = evaluator.get_neighbors(svd_mat, test_words)
#         drift_svd = evaluator.calculate_drift(base_neighbors_svd, curr_neighbors_svd) if base_neighbors_svd else 0.0
#         base_neighbors_svd = curr_neighbors_svd
            
#         sem_svd = max(0, corr_svd)
#         syn_svd = 1.0 - drift_svd
#         sci_svd = 2 * (sem_svd * syn_svd) / (sem_svd + syn_svd) if (sem_svd + syn_svd) > 0 else 0
        
#         sparsity = evaluator.calculate_sparsity(ppmi_mat)
#         optimal_d = svd_mat.shape[1]
        
#         row = {"window": w, "optimal_d": optimal_d, "sparsity_ppmi": sparsity, "corr_ppmi": corr_ppmi, "drift_ppmi": drift_ppmi, "sci_ppmi": sci_ppmi, "corr_svd": corr_svd, "drift_svd": drift_svd, "sci_svd": sci_svd}
#         results.append(row)
#         print(f"  [Result] PPMI SCI: {sci_ppmi:.4f} | SVD SCI: {sci_svd:.4f} | d={optimal_d}")

#     df = pd.DataFrame(results)
#     csv_path = os.path.join(OUTPUT_DIR, "metrics_summary.csv")
#     df.to_csv(csv_path, index=False)
#     print(f"\n[Info] Metrics saved to {csv_path}")
#     save_plots(df, PLOTS_DIR, singular_vals_history)
#     print(f"[Info] Plots saved to {PLOTS_DIR}")

# if __name__ == "__main__":
#     main()

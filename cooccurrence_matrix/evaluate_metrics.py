import os
import numpy as np
import scipy.sparse as sp
import pandas as pd
import requests
import logging
import random
from scipy.stats import spearmanr

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class SemanticEvaluator:
    def __init__(self, vocab):
        self.vocab = vocab
        sample_word = next(iter(vocab))
        self.is_lower_vocab = sample_word.islower() and not sample_word.isupper()
    
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
        
        valid_indices = []
        human_scores = []
        for w1, w2, score in pairs:
            if self.is_lower_vocab:
                w1, w2 = w1.lower(), w2.lower()
            if w1 in self.vocab and w2 in self.vocab:
                valid_indices.append((self.vocab[w1], self.vocab[w2]))
                human_scores.append(score)
        
        if not valid_indices:
            return 0.0, 0
        
        model_scores = []
        for idx1, idx2 in valid_indices:
            if is_sparse:
                sim = (norm_matrix[idx1] @ norm_matrix[idx2].T)[0, 0]
            else:
                sim = float(np.dot(norm_matrix[idx1], norm_matrix[idx2]))
            model_scores.append(sim)
        
        return spearmanr(human_scores, model_scores)[0], len(human_scores)
    
    def evaluate_wordsim(self, matrix, path):
        if not os.path.exists(path):
            logging.warning(f"WordSim path not found: {path}")
            return 0.0, 0
        try:
            df = pd.read_csv(path)
            cols = [c.lower().strip() for c in df.columns]
            df.columns = cols
            w1_col = next((c for c in cols if 'word 1' in c or 'word1' in c), df.columns[0] if len(df.columns)>0 else None)
            w2_col = next((c for c in cols if 'word 2' in c or 'word2' in c), df.columns[1] if len(df.columns)>1 else None)
            score_col = next((c for c in cols if 'human' in c or 'score' in c), df.columns[2] if len(df.columns)>2 else None)
            
            if not all([w1_col, w2_col, score_col]): return 0.0, 0
            
            pairs = []
            for _, row in df.iterrows():
                try: pairs.append((str(row[w1_col]), str(row[w2_col]), float(row[score_col])))
                except: continue
            return self._cosine_sim_batch(matrix, pairs)
        except Exception:
            return 0.0, 0

class AnalogyEvaluator:
    def __init__(self, vocab):
        self.vocab = vocab
        sample_word = next(iter(vocab))
        self.is_lower_vocab = sample_word.islower() and not sample_word.isupper()
    
    # FIX: Added method argument
    def evaluate_analogy(self, matrix, analogy_path, method='3cosadd'):
        if not os.path.exists(analogy_path): return 0.0, 0
        is_sparse = sp.issparse(matrix)
        if is_sparse:
            norms = sp.linalg.norm(matrix, axis=1)
            norms[norms == 0] = 1
            norm_matrix = sp.diags(1.0 / norms) @ matrix
        else:
            norms = np.linalg.norm(matrix, axis=1, keepdims=True)
            norms[norms == 0] = 1
            norm_matrix = matrix / norms
            
        correct, total = 0, 0
        try:
            with open(analogy_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.startswith(('#', ':')): continue
                    parts = line.strip().split()
                    if len(parts) < 4: continue
                    a, a_star, b, target = parts[:4]
                    if self.is_lower_vocab: a, a_star, b, target = a.lower(), a_star.lower(), b.lower(), target.lower()
                    if not all(w in self.vocab for w in [a, a_star, b, target]): continue
                    
                    vecs = norm_matrix[[self.vocab[w] for w in [a, a_star, b]]]
                    if is_sparse: vecs = vecs.toarray()
                    
                    # 3CosAdd: b* = b - a + a*
                    query = vecs[2] - vecs[0] + vecs[1]
                    
                    if is_sparse: scores = (norm_matrix @ query).T
                    else: scores = np.dot(norm_matrix, query)
                    
                    top_idx = np.argsort(scores.flatten())[-4:][::-1]
                    input_ids = {self.vocab[w] for w in [a, a_star, b]}
                    
                    for idx in top_idx:
                        if idx not in input_ids:
                            if idx == self.vocab[target]: correct += 1
                            break
                    total += 1
            return correct/total if total else 0, total
        except Exception: return 0.0, 0

class OutlierDetectionEvaluator:
    def __init__(self, vocab):
        self.vocab = vocab
        sample_word = next(iter(vocab))
        self.is_lower_vocab = sample_word.islower() and not sample_word.isupper()
    
    def evaluate_outlier_detection(self, matrix, outlier_path):
        if not os.path.exists(outlier_path): return 0.0, 0
        
        is_sparse = sp.issparse(matrix)
        if is_sparse:
            norms = sp.linalg.norm(matrix, axis=1)
            norms[norms == 0] = 1
            norm_matrix = sp.diags(1.0 / norms) @ matrix
        else:
            norms = np.linalg.norm(matrix, axis=1, keepdims=True)
            norms[norms == 0] = 1
            norm_matrix = matrix / norms

        results = []
        try:
            with open(outlier_path, 'r', encoding='utf-8') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 9:
                        cluster_words = parts[:-1]
                        target = parts[-1]
                    elif len(parts) == 8:
                        cluster_words = parts
                        target = parts[-1]
                    else:
                        continue
                        
                    if self.is_lower_vocab:
                        cluster_words = [w.lower() for w in cluster_words]
                        target = target.lower()
                    
                    valid_words = [w for w in cluster_words if w in self.vocab]
                    if len(valid_words) < 3: continue 
                    if target not in self.vocab: continue
                    
                    target_idx = self.vocab[target]
                    indices = [self.vocab[w] for w in valid_words]
                    
                    vecs = norm_matrix[indices]
                    if is_sparse: vecs = vecs.toarray()
                    
                    sim_matrix = np.dot(vecs, vecs.T)
                    compactness = []
                    for i in range(len(valid_words)):
                        score = (np.sum(sim_matrix[i]) - 1.0) / (len(valid_words) - 1)
                        compactness.append(score)
                    
                    pred_idx_local = np.argmin(compactness)
                    pred_idx_global = indices[pred_idx_local]
                    results.append(1 if pred_idx_global == target_idx else 0)
                    
            return np.mean(results) if results else 0.0, len(results)
        except Exception as e:
            logging.error(f"Outlier eval error: {e}")
            return 0.0, 0

class StabilityEvaluator:
    def __init__(self, vocab):
        self.vocab = vocab
    def get_neighbors(self, matrix, word_list, k=10): return {}
    def calculate_drift(self, base, curr): return 0.0

class MetricsCalculator:
    def __init__(self, vocab, inverse_vocab):
        self.vocab = vocab
        self.inverse_vocab = inverse_vocab
        self.semantic_eval = SemanticEvaluator(vocab)
        self.analogy_eval = AnalogyEvaluator(vocab)
        self.outlier_eval = OutlierDetectionEvaluator(vocab)
        self.stability_eval = StabilityEvaluator(vocab)

def load_vocab_list(path):
    vocab = {}
    inverse_vocab = {}
    with open(path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            w = line.strip()
            if w:
                vocab[w] = i
                inverse_vocab[i] = w
    return vocab, inverse_vocab

def generate_888_dataset(source_dir, output_file):
    # Same generation logic as before, omitted for brevity but included in full file if needed
    pass 

def main():
    BASE_OUTPUT_DIR = "./data/results4"
    MATRICES_DIR = os.path.join(BASE_OUTPUT_DIR, "matrices")
    DATASETS_DIR = "./data/datasets"
    
    WORDSIM_PATH = os.path.join(DATASETS_DIR, "wordsim353.csv")
    ANALOGY_PATH = os.path.join(DATASETS_DIR, "google_analogy.txt")
    OUTLIER_PATH = os.path.join(DATASETS_DIR, "8-8-8_outliers.txt")
    
    windows = [7]
    for w in windows:
        vocab_file = os.path.join(MATRICES_DIR, f"vocab_pruned_w{w}.txt")
        matrix_file = os.path.join(MATRICES_DIR, f"svd_pruned_w{w}.npy")
        
        if os.path.exists(vocab_file) and os.path.exists(matrix_file):
            print(f"\nEvaluating Pruned Model W={w}...")
            vocab, inverse_vocab = load_vocab_list(vocab_file)
            matrix = np.load(matrix_file)
            
            calc = MetricsCalculator(vocab, inverse_vocab)
            
            ws_corr, ws_cnt = calc.semantic_eval.evaluate_wordsim(matrix, WORDSIM_PATH)
            analogy_acc, an_cnt = calc.analogy_eval.evaluate_analogy(matrix, ANALOGY_PATH, method="3cosadd")
            outlier_acc, out_cnt = calc.outlier_eval.evaluate_outlier_detection(matrix, OUTLIER_PATH)
            
            print(f"  WordSim353: {ws_corr:.4f} (n={ws_cnt})")
            print(f"  Analogy:    {analogy_acc:.4f} (n={an_cnt})")
            print(f"  Outlier:    {outlier_acc:.4f} (n={out_cnt})")

if __name__ == "__main__":
    main()




# import os
# import numpy as np
# import scipy.sparse as sp
# import pandas as pd
# from scipy.stats import spearmanr


# class SemanticEvaluator:
#     def __init__(self, vocab):
#         self.vocab = vocab
    
#     def _cosine_sim_batch(self, matrix, pairs):
#         is_sparse = sp.issparse(matrix)
        
#         if is_sparse:
#             norms = sp.linalg.norm(matrix, axis=1)
#             norms[norms == 0] = 1
#             norm_matrix = sp.diags(1.0 / norms) @ matrix
#         else:
#             norms = np.linalg.norm(matrix, axis=1, keepdims=True)
#             norms[norms == 0] = 1
#             norm_matrix = matrix / norms
        
#         valid_indices = []
#         human_scores = []
#         for w1, w2, score in pairs:
#             if w1 in self.vocab and w2 in self.vocab:
#                 valid_indices.append((self.vocab[w1], self.vocab[w2]))
#                 human_scores.append(score)
        
#         if not valid_indices:
#             return 0.0, 0
        
#         model_scores = []
#         for idx1, idx2 in valid_indices:
#             if is_sparse:
#                 sim = (norm_matrix[idx1] @ norm_matrix[idx2].T)[0, 0]
#             else:
#                 sim = np.dot(norm_matrix[idx1], norm_matrix[idx2])
#             model_scores.append(sim)
        
#         return spearmanr(human_scores, model_scores)[0], len(human_scores)
    
#     def evaluate_wordsim(self, matrix, path):
#         if not os.path.exists(path):
#             return 0.0, 0
        
#         df = pd.read_csv(path)
#         cols = [c.lower() for c in df.columns]
#         w1_idx = next(i for i, c in enumerate(cols) if 'word 1' in c or 'word1' in c)
#         w2_idx = next(i for i, c in enumerate(cols) if 'word 2' in c or 'word2' in c)
#         s_idx = next(i for i, c in enumerate(cols) if 'human' in c or 'score' in c)
        
#         pairs = []
#         for row in df.itertuples(index=False):
#             pairs.append((str(row[w1_idx]), str(row[w2_idx]), float(row[s_idx])))
        
#         return self._cosine_sim_batch(matrix, pairs)


# class StabilityEvaluator:
#     def __init__(self, vocab):
#         self.vocab = vocab
    
#     def get_neighbors(self, matrix, word_list, k=10):
#         neighbors = {}
#         is_sparse = sp.issparse(matrix)
        
#         if is_sparse:
#             norms = sp.linalg.norm(matrix, axis=1)
#             norms[norms == 0] = 1
#             norm_matrix = sp.diags(1.0 / norms) @ matrix
#         else:
#             norms = np.linalg.norm(matrix, axis=1, keepdims=True)
#             norms[norms == 0] = 1
#             norm_matrix = matrix / norms
        
#         for word in word_list:
#             if word not in self.vocab:
#                 continue
#             idx = self.vocab[word]
#             if is_sparse:
#                 scores = (norm_matrix[idx] @ norm_matrix.T).toarray().flatten()
#             else:
#                 scores = np.dot(norm_matrix[idx], norm_matrix.T)
#             top_k = np.argsort(scores)[-(k + 1):][::-1]
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
#                 else:
#                     drifts.append(1.0)
#         return np.mean(drifts) if drifts else 0.0


# class MetricsCalculator:
#     def __init__(self, vocab, inverse_vocab):
#         self.vocab = vocab
#         self.inverse_vocab = inverse_vocab
#         self.semantic_eval = SemanticEvaluator(vocab)
#         self.stability_eval = StabilityEvaluator(vocab)
    
#     def calculate_sci(self, semantic, stability):
#         sem = max(0, semantic)
#         syn = 1.0 - stability
#         if (sem + syn) > 0:
#             return 2 * (sem * syn) / (sem + syn)
#         return 0
    
#     def calculate_sparsity(self, matrix):
#         if sp.issparse(matrix):
#             return matrix.nnz / (matrix.shape[0] * matrix.shape[1])
#         return 1.0


# def load_vocab(path, top_k=25000):
#     vocab = {}
#     inverse_vocab = {}
    
#     raw_counts = []
#     with open(path, "r", encoding="utf-8") as f:
#         for line in f:
#             parts = line.strip().split('\t')
#             if len(parts) >= 2:
#                 word = parts[1]
#                 count = int(parts[2]) if len(parts) >= 3 else 1
#                 raw_counts.append((word, count))
    
#     raw_counts.sort(key=lambda x: x[1], reverse=True)
#     truncated = raw_counts[:top_k]
    
#     for idx, (word, _) in enumerate(truncated):
#         vocab[word] = idx
#         inverse_vocab[idx] = word
    
#     return vocab, inverse_vocab


# def main():
#     BASE_PATH = "./data/corpora/eng_wikipedia_2016_1M"
#     WORDS_PATH = os.path.join(BASE_PATH, "eng_wikipedia_2016_1M-words.txt")
#     WORDSIM_PATH = "./data/wordsim353.csv"
#     MATRICES_DIR = "./data/results3/matrices"
#     OUTPUT_DIR = "./data/results3"
    
#     os.makedirs(OUTPUT_DIR, exist_ok=True)
    
#     print("Loading vocabulary...")
#     vocab, inverse_vocab = load_vocab(WORDS_PATH, top_k=30000)
    
#     test_words = [inverse_vocab[i] for i in range(min(100, len(inverse_vocab)))]
    
#     calculator = MetricsCalculator(vocab, inverse_vocab)
#     windows = [5, 6, 7, 8, 9, 10]
    
#     results = []
#     base_neighbors_ppmi = None
#     base_neighbors_svd = None
    
#     for w in windows:
#         print(f"\nEvaluating window size: {w}")
        
#         ppmi_path = os.path.join(MATRICES_DIR, f"ppmi_w{w}.npz")
#         svd_path = os.path.join(MATRICES_DIR, f"svd_w{w}.npy")
        
#         if not os.path.exists(ppmi_path) or not os.path.exists(svd_path):
#             print(f"Missing matrices for window {w}")
#             continue
        
#         ppmi_mat = sp.load_npz(ppmi_path)
#         svd_mat = np.load(svd_path)
        
#         corr_ppmi, _ = calculator.semantic_eval.evaluate_wordsim(ppmi_mat, WORDSIM_PATH)
#         curr_neighbors_ppmi = calculator.stability_eval.get_neighbors(ppmi_mat, test_words)
#         drift_ppmi = calculator.stability_eval.calculate_drift(base_neighbors_ppmi, curr_neighbors_ppmi) if base_neighbors_ppmi else 0.0
#         base_neighbors_ppmi = curr_neighbors_ppmi
#         sci_ppmi = calculator.calculate_sci(corr_ppmi, drift_ppmi)
        
#         corr_svd, _ = calculator.semantic_eval.evaluate_wordsim(svd_mat, WORDSIM_PATH)
#         curr_neighbors_svd = calculator.stability_eval.get_neighbors(svd_mat, test_words)
#         drift_svd = calculator.stability_eval.calculate_drift(base_neighbors_svd, curr_neighbors_svd) if base_neighbors_svd else 0.0
#         base_neighbors_svd = curr_neighbors_svd
#         sci_svd = calculator.calculate_sci(corr_svd, drift_svd)
        
#         sparsity = calculator.calculate_sparsity(ppmi_mat)
#         optimal_d = svd_mat.shape[1]
        
#         row = {
#             "window": w,
#             "optimal_d": optimal_d,
#             "sparsity_ppmi": sparsity,
#             "corr_ppmi": corr_ppmi,
#             "drift_ppmi": drift_ppmi,
#             "sci_ppmi": sci_ppmi,
#             "corr_svd": corr_svd,
#             "drift_svd": drift_svd,
#             "sci_svd": sci_svd
#         }
#         results.append(row)
#         print(f"  PPMI SCI: {sci_ppmi:.4f} | SVD SCI: {sci_svd:.4f}")
    
#     df = pd.DataFrame(results)
#     csv_path = os.path.join(OUTPUT_DIR, "metrics_summary.csv")
#     df.to_csv(csv_path, index=False)
#     print(f"\nMetrics saved to {csv_path}")


# if __name__ == "__main__":
#     main()

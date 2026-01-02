import os
import numpy as np
import pandas as pd
from evaluate_metrics import MetricsCalculator

class PretrainedEmbeddingLoader:
    @staticmethod
    def load_fasttext(path, max_vocab=None):
        print(f"Loading FastText embeddings from {path}...")
        vocab = {}
        inverse_vocab = {}
        embeddings_list = []
        
        with open(path, 'r', encoding='utf-8', errors='ignore') as f:
            # Parse header
            first_line = next(f).strip().split()
            if len(first_line) == 2:
                try:
                    n_words = int(first_line[0])
                    dim = int(first_line[1])
                    print(f"File header: {n_words} words, {dim} dimensions")
                except ValueError:
                    # Not a header, reset
                    f.seek(0)
                    dim = None
            else:
                f.seek(0)
                dim = None
            
            idx = 0
            for line in f:
                parts = line.rstrip().split()
                if not parts: continue
                
                # Heuristic to find split between word and vector
                # If we know dim, use it. If not, guess based on length.
                if dim is not None:
                    if len(parts) <= dim: continue # Line too short
                    # FastText can have spaces in words
                    word = " ".join(parts[:-dim])
                    vec_parts = parts[-dim:]
                else:
                    
                    # Assuming standard 300d
                    word = parts[0]
                    vec_parts = parts[1:]
                    if dim is None:
                        dim = len(vec_parts) # Set dim from first line
                
                try:
                    vec = np.array([float(x) for x in vec_parts], dtype=np.float32)
                except ValueError:
                    continue
                
                # Double check dimension
                if len(vec) != dim:
                    continue
                    
                vocab[word] = idx
                inverse_vocab[idx] = word
                embeddings_list.append(vec)
                idx += 1
                
                if max_vocab and idx >= max_vocab:
                    break
                
                if idx % 100000 == 0:
                    print(f"  Loaded {idx} words...")
        
        embeddings = np.stack(embeddings_list)
        print(f"Loaded embeddings: {embeddings.shape}")
        return embeddings, vocab, inverse_vocab

def main():
    WORDSIM_PATH = "./data/datasets/wordsim353.csv"
    GOOGLE_ANALOGY_PATH = "./data/datasets/google_analogy.txt"
    OUTLIER_PATH = "./data/datasets/8-8-8_outliers.txt"
    
    PRETRAINED_PATH = "./wiki.en.vec"
    OUTPUT_DIR = "./data/results_pretrained"
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    loader = PretrainedEmbeddingLoader()
    
    # Load fasttext
    embeddings, vocab, inverse_vocab = loader.load_fasttext(PRETRAINED_PATH, max_vocab=30000)
    
    calculator = MetricsCalculator(vocab, inverse_vocab)
    
    print("\nEvaluating Pretrained...")
    
    results = {
        "embedding_type": "fasttext",
        "vocab_size": len(vocab),
        "dim": embeddings.shape[1]
    }
    
    # WordSim
    (ws_corr, ws_pval), ws_cnt = calculator.semantic_eval.evaluate_wordsim(embeddings, WORDSIM_PATH)

    results["wordsim353_rho"] = ws_corr
    results["wordsim353_pval"] = ws_pval
    results["wordsim_n"] = ws_cnt
    print(f"WordSim353: rho={ws_corr:.4f}, p={ws_pval:.4g} (n={ws_cnt})")
    
    # Analogy
    acc, n = calculator.analogy_eval.evaluate_analogy(embeddings, GOOGLE_ANALOGY_PATH, method="3cosadd")
    results["analogy_acc"] = acc
    results["analogy_n"] = n
    print(f"Analogy:    {acc:.4f} (n={n})")
    
    # Outlier
    if os.path.exists(OUTLIER_PATH):
        out_acc, out_n = calculator.outlier_eval.evaluate(embeddings, OUTLIER_PATH)
        results["outlier_acc"] = out_acc
        results["outlier_n"] = out_n
        print(f"Outlier:    {out_acc:.4f} (n={out_n})")
    else:
        print("Outlier file not found.")
        
    # Save
    pd.DataFrame([results]).to_csv(os.path.join(OUTPUT_DIR, "metrics_fasttext.csv"), index=False)
    print(f"Saved results to {OUTPUT_DIR}")

if __name__ == "__main__":
    main()

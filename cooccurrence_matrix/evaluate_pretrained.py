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
                    # FastText can have spaces in words? usually not in .vec, but let's be safe
                    # Standard .vec is "word dim1 dim2 ..."
                    word = " ".join(parts[:-dim])
                    vec_parts = parts[-dim:]
                else:
                    # Guess: last 300 (or similar) are vector. 
                    # Assuming standard 300d if unknown, or infer from first valid line
                    # Let's try to infer dim from first line if unknown
                    # This branch is risky without a header, let's assume standard behavior:
                    # Word is part[0], rest is vec
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
    
    # 1. WordSim
    corr, n_pairs = calculator.semantic_eval.evaluate_wordsim(embeddings, WORDSIM_PATH)
    results["wordsim353"] = corr
    results["wordsim_n"] = n_pairs
    print(f"WordSim353: {corr:.4f} (n={n_pairs})")
    
    # 2. Analogy
    acc, n = calculator.analogy_eval.evaluate_analogy(embeddings, GOOGLE_ANALOGY_PATH, method="3cosadd")
    results["analogy_acc"] = acc
    results["analogy_n"] = n
    print(f"Analogy:    {acc:.4f} (n={n})")
    
    # 3. Outlier
    if os.path.exists(OUTLIER_PATH):
        out_acc, out_n = calculator.outlier_eval.evaluate_outlier_detection(embeddings, OUTLIER_PATH)
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



# import os
# import numpy as np
# import pandas as pd
# from evaluate_metrics import SemanticEvaluator, StabilityEvaluator, MetricsCalculator


# class PretrainedEmbeddingLoader:
#     @staticmethod
#     def load_fasttext(path, max_vocab=None):
#         print(f"Loading FastText embeddings from {path}...")
#         vocab = {}
#         inverse_vocab = {}
#         embeddings_list = []
        
#         with open(path, 'r', encoding='utf-8') as f:
#             first_line = next(f)
#             parts = first_line.strip().split()
#             if len(parts) == 2:
#                 total_words, dim = int(parts[0]), int(parts[1])
#                 print(f"File header: {total_words} words, {dim} dimensions")
#             else:
#                 f.seek(0)
#                 dim = None
            
#             idx = 0
#             for line in f:
#                 parts = line.strip().split()
#                 if len(parts) < 10:
#                     continue
                
#                 try:
#                     if dim is None:
#                         dim = len(parts) - 1
                    
#                     if len(parts) != dim + 1:
#                         word = ' '.join(parts[:-dim])
#                         vec = np.array([float(x) for x in parts[-dim:]], dtype=np.float32)
#                     else:
#                         word = parts[0]
#                         vec = np.array([float(x) for x in parts[1:]], dtype=np.float32)
                    
#                     if len(vec) != dim:
#                         continue
                    
#                     vocab[word] = idx
#                     inverse_vocab[idx] = word
#                     embeddings_list.append(vec)
#                     idx += 1
                    
#                     if max_vocab and idx >= max_vocab:
#                         break
                    
#                     if idx % 100000 == 0:
#                         print(f"  Loaded {idx} words...")
                        
#                 except (ValueError, IndexError):
#                     continue
        
#         embeddings = np.array(embeddings_list, dtype=np.float32)
#         print(f"Loaded embeddings for {len(vocab)} words, dimension {embeddings.shape[1]}")
        
#         return embeddings, vocab, inverse_vocab
    
#     # @staticmethod
#     # def load_word2vec(path, max_vocab=None):
#     #     print(f"Loading Word2Vec embeddings from {path}...")
#     #     from gensim.models import KeyedVectors
#     #     model = KeyedVectors.load_word2vec_format(path, binary=True)
        
#     #     vocab = {}
#     #     inverse_vocab = {}
#     #     embeddings_list = []
        
#     #     for idx, word in enumerate(model.index_to_key):
#     #         if max_vocab and idx >= max_vocab:
#     #             break
#     #         vocab[word] = idx
#     #         inverse_vocab[idx] = word
#     #         embeddings_list.append(model[word])
            
#     #         if idx % 100000 == 0 and idx > 0:
#     #             print(f"  Loaded {idx} words...")
        
#     #     embeddings = np.array(embeddings_list, dtype=np.float32)
#     #     print(f"Loaded embeddings for {len(vocab)} words, dimension {embeddings.shape[1]}")
        
#     #     return embeddings, vocab, inverse_vocab
    
#     # @staticmethod
#     # def load_glove(path, max_vocab=None):
#     #     print(f"Loading GloVe embeddings from {path}...")
#     #     vocab = {}
#     #     inverse_vocab = {}
#     #     embeddings_list = []
#     #     dim = None
        
#     #     with open(path, 'r', encoding='utf-8') as f:
#     #         idx = 0
#     #         for line in f:
#     #             parts = line.strip().split()
#     #             if len(parts) < 10:
#     #                 continue
                
#     #             try:
#     #                 if dim is None:
#     #                     dim = len(parts) - 1
                    
#     #                 if len(parts) != dim + 1:
#     #                     word = ' '.join(parts[:-dim])
#     #                     vec = np.array([float(x) for x in parts[-dim:]], dtype=np.float32)
#     #                 else:
#     #                     word = parts[0]
#     #                     vec = np.array([float(x) for x in parts[1:]], dtype=np.float32)
                    
#     #                 if len(vec) != dim:
#     #                     continue
                    
#     #                 vocab[word] = idx
#     #                 inverse_vocab[idx] = word
#     #                 embeddings_list.append(vec)
#     #                 idx += 1
                    
#     #                 if max_vocab and idx >= max_vocab:
#     #                     break
                    
#     #                 if idx % 100000 == 0:
#     #                     print(f"  Loaded {idx} words...")
                        
#     #             except (ValueError, IndexError):
#     #                 continue
        
#     #     embeddings = np.array(embeddings_list, dtype=np.float32)
#     #     print(f"Loaded embeddings for {len(vocab)} words, dimension {embeddings.shape[1]}")
        
#     #     return embeddings, vocab, inverse_vocab


# def main():
#     WORDSIM_PATH = "./data/wordsim353.csv"
#     PRETRAINED_PATH = "./wiki.en.vec"
#     OUTPUT_DIR = "./data/results3/pretrained"
    
#     os.makedirs(OUTPUT_DIR, exist_ok=True)
    
#     loader = PretrainedEmbeddingLoader()
    
#     embedding_type = "fasttext"
#     max_vocab = None
    
#     print(f"Loading {embedding_type} embeddings (max {max_vocab} words)...")
    
#     if embedding_type == "fasttext":
#         embeddings, vocab, inverse_vocab = loader.load_fasttext(PRETRAINED_PATH, max_vocab=max_vocab)
#     elif embedding_type == "word2vec":
#         embeddings, vocab, inverse_vocab = loader.load_word2vec(PRETRAINED_PATH, max_vocab=max_vocab)
#     elif embedding_type == "glove":
#         embeddings, vocab, inverse_vocab = loader.load_glove(PRETRAINED_PATH, max_vocab=max_vocab)
#     else:
#         raise ValueError(f"Unknown embedding type: {embedding_type}")
    
#     print(f"Embedding shape: {embeddings.shape}")
    
#     calculator = MetricsCalculator(vocab, inverse_vocab)
    
#     print("\nEvaluating semantic quality...")
#     corr, n_pairs = calculator.semantic_eval.evaluate_wordsim(embeddings, WORDSIM_PATH)
#     print(f"WordSim353 Spearman correlation: {corr:.4f} ({n_pairs} pairs)")
    
#     test_words = [inverse_vocab[i] for i in range(min(100, len(inverse_vocab)))]
#     neighbors = calculator.stability_eval.get_neighbors(embeddings, test_words, k=10)
#     print(f"Computed neighbors for {len(neighbors)} test words")
    
#     results = {
#         "embedding_type": embedding_type,
#         "embedding_path": PRETRAINED_PATH,
#         "vocab_size": len(vocab),
#         "embedding_dim": embeddings.shape[1],
#         "wordsim_correlation": corr,
#         "wordsim_pairs": n_pairs
#     }
    
#     df = pd.DataFrame([results])
#     csv_path = os.path.join(OUTPUT_DIR, f"metrics_{embedding_type}.csv")
#     df.to_csv(csv_path, index=False)
#     print(f"\nMetrics saved to {csv_path}")
    
#     embeddings_save_path = os.path.join(OUTPUT_DIR, f"embeddings_{embedding_type}.npy")
#     np.save(embeddings_save_path, embeddings)
#     print(f"Embeddings saved to {embeddings_save_path}")


# if __name__ == "__main__":
#     main()

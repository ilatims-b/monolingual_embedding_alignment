# import os
# import numpy as np
# import scipy.sparse as sp
# import csv

# def load_vocab(path, top_k=30000):
#     """
#     Re-uses logic to load vocab to ensure indices match the matrix.
#     """
#     print(f"Loading vocabulary from {path} (Top-{top_k})...")
#     if not os.path.exists(path):
#         raise FileNotFoundError(f"Vocab file not found: {path}")
        
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
    
#     # We only need the list of words ordered by index
#     ordered_words = [word for word, _ in truncated]
#     return ordered_words

# def export_matrix(matrix, vocab_list, output_prefix, max_rows=10000):
#     """
#     Exports matrix and vocab to .tsv files for TensorFlow Projector.
    
#     Args:
#         matrix: numpy array or scipy sparse matrix (N, D)
#         vocab_list: list of strings (N,)
#         output_prefix: path prefix for output files (e.g., "data/projector/svd_w5")
#         max_rows: limit number of rows to export to prevent browser crash (default 10k)
#     """
    
#     # Ensure directory exists
#     os.makedirs(os.path.dirname(output_prefix), exist_ok=True)
    
#     meta_path = f"{output_prefix}_metadata.tsv"
#     vecs_path = f"{output_prefix}_vectors.tsv"
    
#     # 1. Determine subset to export
#     # Projector can handle ~10k-50k points comfortably. 
#     # Full vocab (e.g., 30k) might be slow but okay. 100k+ will crash.
#     n_rows = min(matrix.shape[0], max_rows, len(vocab_list))
    
#     print(f"Exporting top {n_rows} vectors to:")
#     print(f"  Metadata: {meta_path}")
#     print(f"  Vectors:  {vecs_path}")
    
#     # 2. Write Metadata (Words)
#     with open(meta_path, 'w', encoding='utf-8', newline='') as f:
#         for i in range(n_rows):
#             f.write(f"{vocab_list[i]}\n")
            
#     # 3. Write Vectors
#     # Handle Sparse Matrix (PPMI)
#     if sp.issparse(matrix):
#         print("  [Info] Converting sparse matrix to dense for export (this may be slow)...")
#         # Slicing a CSR matrix is efficient
#         sub_matrix = matrix[:n_rows].toarray()
#     else:
#         sub_matrix = matrix[:n_rows]
        
#     # Using numpy savetxt is efficient for writing TSVs
#     np.savetxt(vecs_path, sub_matrix, delimiter='\t', fmt='%.5f')
    
#     print("  Done.")


# def main():
#     # --- CONFIGURATION ---
#     BASE_PATH = "./data/corpora/eng_wikipedia_2016_1M"
#     WORDS_PATH = os.path.join(BASE_PATH, "eng_wikipedia_2016_1M-words.txt")
#     MATRICES_DIR = "./data/results4/matrices"
#     OUTPUT_DIR = "./data/projector_files"
#     ppmiMATRICES_DIR = "./data/results3/matrices"
#     # Set the vocab size used during matrix creation
#     VOCAB_SIZE = 30000 
    
#     # How many top frequent words to visualize?
#     # Keeping it under 10k makes the projector smoother.
#     EXPORT_LIMIT = 30000
    
#     # 1. Load Vocab
#     vocab_list = load_vocab(WORDS_PATH, top_k=VOCAB_SIZE)
    
#     # 2. Select window size to export
#     # You can loop through multiple windows if desired
#     target_window = 7
    
#     # --- EXPORT SVD (Recommended) ---
#     svd_path = os.path.join(MATRICES_DIR, f"svd_pruned_w{target_window}.npy")
#     if os.path.exists(svd_path):
#         print(f"\nProcessing SVD Matrix (W={target_window})...")
#         svd_mat = np.load(svd_path)
#         export_matrix(svd_mat, vocab_list, 
#                       os.path.join(OUTPUT_DIR, f"svd_pruned_w{target_window}"), 
#                       max_rows=EXPORT_LIMIT)
#     else:
#         print(f"SVD matrix for w={target_window} not found.")

#     #plot the distribution of norm of the embeddigns
#     norms = np.linalg.norm(svd_mat, axis=1)
#     import matplotlib.pyplot as plt
#     plt.figure(figsize=(8,6))
#     plt.hist(norms, bins=50, color='blue', alpha=0.7)
#     plt.title(f'Norm Distribution of SVD Embeddings (W={target_window})')
#     plt.xlabel('Norm Value')
#     plt.ylabel('Frequency')
#     plt.grid(True)
#     plt_path = os.path.join(OUTPUT_DIR, f'svd_pruned_w{target_window}_norm_distribution.png')
#     plt.savefig(plt_path)
#     print(f"Norm distribution plot saved to {plt_path}")
#     plt.close()

#     #print some zero norm words sampled randomly from all the zero normed words
#     zero_norm_indices = np.where(norms == 0)[0]
#     if len(zero_norm_indices) > 0:
#         print(f"\nFound {len(zero_norm_indices)} zero-norm embeddings. Sample words:")
#         sample_size = min(20, len(zero_norm_indices))
#         sampled_indices = np.random.choice(zero_norm_indices, size=sample_size, replace=False)
#         for idx in sampled_indices:
#             print(f"  Index: {idx}, Word: {vocab_list[idx]}")
#     else:
#         print("\nNo zero-norm embeddings found.")
#     #get the ppmi matrix of this svd matrix, find the row_sums and calculate the number of rows with row_sums==0 are these same as the ones with zero norm in svd? find the intersection and print some of these words
#     ppmi_path = os.path.join(ppmiMATRICES_DIR, f"ppmi_w{target_window}.npz")
#     if os.path.exists(ppmi_path):
#         ppmi_mat = sp.load_npz(ppmi_path)
#         row_sums = np.array(ppmi_mat.sum(axis=1)).flatten()
#         zero_row_sum_indices = np.where(row_sums == 0)[0]
#         intersection = set(zero_norm_indices).intersection(set(zero_row_sum_indices))
#         print(f"\nNumber of embeddings with zero norm and zero row sum in PPMI: {len(intersection)}. Sample words:")
#         sample_size = min(20, len(intersection))
#         sampled_indices = np.random.choice(list(intersection), size=sample_size, replace=False)
#         for idx in sampled_indices:
#             print(f"  Index: {idx}, Word: {vocab_list[idx]}")
#     else:
#         print(f"PPMI matrix for w={target_window} not found, cannot compare zero row sums.")



#     # --- EXPORT PPMI (Optional - High Dimension!) ---
#     # Warning: PPMI is usually vocab_size x vocab_size (30k x 30k).
#     # Exporting this to TSV will result in a HUGE file (GBs).
#     # Only do this if you really need raw PPMI visualization.
#     # We will limit columns for PPMI to make it feasible (e.g., top 2000 context words).
    
#     # ppmi_path = os.path.join(MATRICES_DIR, f"ppmi_w{target_window}.npz")
#     # if os.path.exists(ppmi_path):
#     #     print(f"\nProcessing PPMI Matrix (W={target_window})...")
#     #     ppmi_mat = sp.load_npz(ppmi_path)
        
#     #     # Truncate columns (dimensions) for PPMI because 30k dimensions is too big for TSV
#     #     # We'll take the top 500 columns (most frequent context words)
#     #     ppmi_dim_limit = 500 
#     #     print(f"  [Info] Truncating PPMI dimensions to top {ppmi_dim_limit} columns for feasibility.")
#     #     ppmi_mat_truncated = ppmi_mat[:, :ppmi_dim_limit]
        
#     #     export_matrix(ppmi_mat_truncated, vocab_list, 
#     #                   os.path.join(OUTPUT_DIR, f"ppmi_w{target_window}"), 
#     #                   max_rows=EXPORT_LIMIT)
#     # else:
#     #     print(f"PPMI matrix for w={target_window} not found.")

# if __name__ == "__main__":
#     main()
import os
import numpy as np
import scipy.sparse as sp
import matplotlib.pyplot as plt


def load_pruned_vocab(vocab_path):
    with open(vocab_path, "r", encoding="utf-8") as f:
        words = [line.strip() for line in f]
    return words


def export_matrix(matrix, vocab_list, output_prefix, max_rows=10000):
    """
    Exports:
      - output_prefix_metadata.tsv         (unchanged)
      - output_prefix_vectors.tsv          (raw embeddings)
      - output_prefix_vectors_norm.tsv     (row‑normalized embeddings)
    """
    os.makedirs(os.path.dirname(output_prefix), exist_ok=True)

    meta_path = f"{output_prefix}_metadata.tsv"
    vecs_path = f"{output_prefix}_vectors.tsv"
    vecs_norm_path = f"{output_prefix}_vectors_norm.tsv"

    n_rows = min(matrix.shape[0], max_rows, len(vocab_list))
    print(f"Exporting {n_rows} vectors to:")
    print(f"  Metadata:        {meta_path}")
    print(f"  Raw vectors:     {vecs_path}")
    print(f"  Normalized vecs: {vecs_norm_path}")

    # metadata
    with open(meta_path, "w", encoding="utf-8") as f:
        for i in range(n_rows):
            f.write(vocab_list[i] + "\n")

    # slice matrix (dense)
    if sp.issparse(matrix):
        sub_matrix = matrix[:n_rows].toarray()
    else:
        sub_matrix = matrix[:n_rows]

    # save raw vectors
    np.savetxt(vecs_path, sub_matrix, delimiter="\t", fmt="%.5f")

    # L2-normalize rows and save normalized vectors
    norms = np.linalg.norm(sub_matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    sub_matrix_norm = sub_matrix / norms
    np.savetxt(vecs_norm_path, sub_matrix_norm, delimiter="\t", fmt="%.5f")

    print("  Done.")


def main():
    MATRICES_DIR = "./data/results4/matrices"
    OUTPUT_DIR = "./data/projector_files"
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    target_window = 7

    svd_path = os.path.join(MATRICES_DIR, f"svd_pruned_w{target_window}.npy")
    vocab_path = os.path.join(MATRICES_DIR, f"vocab_pruned_w{target_window}.txt")

    if not os.path.exists(svd_path):
        raise FileNotFoundError(svd_path)
    if not os.path.exists(vocab_path):
        raise FileNotFoundError(vocab_path)

    svd_mat = np.load(svd_path)
    vocab_pruned = load_pruned_vocab(vocab_path)
    assert svd_mat.shape[0] == len(vocab_pruned), "rows vs vocab mismatch"

    # export both raw and normalized embeddings
    export_matrix(
        svd_mat,
        vocab_pruned,
        os.path.join(OUTPUT_DIR, f"svd_pruned_w{target_window}"),
        max_rows=30000,
    )

    norms = np.linalg.norm(svd_mat, axis=1)
    plt.figure(figsize=(8, 6))
    plt.hist(norms, bins=50, color="blue", alpha=0.7)
    plt.title(f"Norm Distribution of PRUNED SVD Embeddings (W={target_window})")
    plt.xlabel("Norm Value")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt_path = os.path.join(OUTPUT_DIR, f"svd_pruned_w{target_window}_norm_distribution.png")
    plt.savefig(plt_path)
    plt.close()
    print(f"Norm distribution plot saved to {plt_path}")


if __name__ == "__main__":
    main()

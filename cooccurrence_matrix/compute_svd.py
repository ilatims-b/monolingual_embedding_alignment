import os
import numpy as np
import scipy.sparse as sp
from sklearn.decomposition import TruncatedSVD


class SVDComputer:
    def __init__(self, fixed_d=300):
        self.fixed_d = fixed_d
    
    def compute(self, matrix, output_path, singular_vals_path):
        if os.path.exists(output_path) and os.path.exists(singular_vals_path):
            print(f"Loading cached SVD from {output_path}")
            return np.load(output_path), np.load(singular_vals_path)
        
        k = min(self.fixed_d, matrix.shape[1] - 1)
        print(f"Computing SVD (d={k})...")
        
        svd = TruncatedSVD(n_components=k, n_iter=5, random_state=42)
        embeddings = svd.fit_transform(matrix)
        s_vals = svd.singular_values_
        
        np.save(output_path, embeddings)
        np.save(singular_vals_path, s_vals)
        print(f"SVD embeddings saved to {output_path}")
        return embeddings, s_vals


def main():
    MATRICES_DIR = "./data/results3/matrices"
    windows = [5, 6, 7, 8, 9, 10]
    
    svd_computer = SVDComputer(fixed_d=300)
    
    for w in windows:
        print(f"\nProcessing window size: {w}")
        
        ppmi_path = os.path.join(MATRICES_DIR, f"ppmi_w{w}.npz")
        svd_path = os.path.join(MATRICES_DIR, f"svd_w{w}.npy")
        singular_vals_path = os.path.join(MATRICES_DIR, f"singular_values_w{w}.npy")
        
        if not os.path.exists(ppmi_path):
            print(f"PPMI matrix not found: {ppmi_path}")
            continue
        
        ppmi_mat = sp.load_npz(ppmi_path)
        embeddings, s_vals = svd_computer.compute(ppmi_mat, svd_path, singular_vals_path)
        print(f"Embedding shape: {embeddings.shape}")
    
    print("\nSVD computation complete.")


if __name__ == "__main__":
    main()

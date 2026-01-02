import numpy as np
import scipy.linalg
import scipy.sparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
import ot
import pandas as pd
import matplotlib.pyplot as plt
import requests
import shutil
from typing import Optional, Tuple, List, Dict, Set, Any


def get_device():
    """Automatically selects the best available device."""
    if torch.cuda.is_available():
        print("Device: CUDA (NVIDIA GPU)")
        return torch.device('cuda')
    elif torch.backends.mps.is_available():
        print("Device: MPS (Apple Silicon/Metal)")
        return torch.device('mps')
    else:
        print("Device: CPU")
        return torch.device('cpu')

class DataHandler:
    def __init__(self, data_dir="data_alignment50k_hindi_bisparse"):
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)
        self.matrices_dir = os.path.join(data_dir, "matrices")
        self.plots_dir = os.path.join(data_dir, "plots")
        os.makedirs(self.matrices_dir, exist_ok=True)
        os.makedirs(self.plots_dir, exist_ok=True)

    def load_vectors(self, path: str, n_max: Optional[int] = None) -> Tuple[Dict[str, np.ndarray], List[str], np.ndarray]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"{path} not found.")
            
        words = []
        matrix_list = []
        
        print(f"Loading {path}...")
        with open(path, 'r', encoding='utf-8', errors='ignore') as f:
            first=f.readline().strip().split()
            has_header = len(first) == 2
            if has_header:
                pass
            else:
                f.seek(0)
            for i, line in enumerate(f):
                if n_max is not None and i >= n_max: break
                parts = line.rstrip().split(' ')
                word = parts[0].lower()
                vec = np.array(parts[1:], dtype=np.float32)
                matrix_list.append(vec)
                words.append(word)

        matrix = np.array(matrix_list)
        print(f"  Centering {len(matrix)} vectors...")
        matrix -= np.mean(matrix, axis=0)
        
        print("  Normalizing vectors...")
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        matrix /= (norms + 1e-9)
        
        # Build Dictionary Map
        vectors = {w: matrix[i] for i, w in enumerate(words)}
        
        return vectors, words, matrix

    def load_dictionary(self, path: str) -> List[Tuple[str, str]]:
        pairs = []
        if not os.path.exists(path):
            print(f"Warning: Dictionary {path} not found.")
            return []
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split()
                if parts[0] != parts[0].lower():
                    print (f"Non-lowercase word in dict: {parts[0]}")
                if parts[1] != parts[1].lower():
                    print (f"Non-lowercase word in dict: {parts[1]}")    
                if len(parts) >= 2:
                    pairs.append((parts[0], parts[1]))
        return pairs
    
    def save_embedding(self, name, mat):
        np.save(os.path.join(self.matrices_dir, f"{name}.npy"), mat)


# 2. Evaluation Metrics

class Evaluator:
    @staticmethod
    def get_csls_sim(source_emb, target_emb, k=10):
        sim_mat = np.dot(source_emb, target_emb.T)
        ra_src = np.sort(sim_mat, axis=1)[:, -k:]
        mean_ra_src = np.mean(ra_src, axis=1)
        ra_tgt = np.sort(sim_mat, axis=0)[-k:, :]
        mean_ra_tgt = np.mean(ra_tgt, axis=0)
        csls_sim = 2 * sim_mat - mean_ra_src[:, None] - mean_ra_tgt[None, :]
        return csls_sim

    @staticmethod
    def bal_ap_inc(src_vecs, tgt_vecs, k: Optional[int] = None, eps: float = 1e-12):
        """
        balAPinc(u->v) = sqrt( LIN(u,v) * APinc(u->v) )
        - Feature lists: ranked nonzero dimensions by weight (descending).
        - Membership for APinc: feature is relevant iff it's NONZERO in v (not "in top-k of v").
        This matches the paper's use of ranked feature lists and rel'(f)=1 iff f in FV_v. [page:0]
        """
        if scipy.sparse.issparse(src_vecs): src_vecs = src_vecs.toarray()
        if scipy.sparse.issparse(tgt_vecs): tgt_vecs = tgt_vecs.toarray()

        n_pairs = src_vecs.shape[0]
        scores = []

        for i in range(n_pairs):
            u = src_vecs[i]
            v = tgt_vecs[i]

            # Nonzero feature indices
            u_nz = np.flatnonzero(u > 0)
            v_nz = np.flatnonzero(v > 0)
            if u_nz.size == 0 or v_nz.size == 0:
                scores.append(0.0)
                continue

            # Ranked feature list FVu (descending weights)
            u_rank = u_nz[np.argsort(-u[u_nz])]
            if k is not None:
                u_rank = u_rank[:k]

            v_set = set(v_nz.tolist())  # FVv as a set (NONZERO membership)

            # LIN(u,v) from Eq (1), using nonzero sets [page:0]
            inter = np.intersect1d(u_nz, v_nz, assume_unique=False)
            lin_num = float(np.sum(u[inter]) + np.sum(v[inter]))
            lin_den = float(np.sum(u[u_nz]) + np.sum(v[v_nz]) + eps)
            LIN = lin_num / lin_den

            # APinc(u->v) from Eq (2) with rel'(f)=1 iff f in FVv [page:0]
            hits = 0
            ap_sum = 0.0
            for r, feat in enumerate(u_rank, start=1):
                if feat in v_set:
                    hits += 1
                    ap_sum += hits / r  # P@r contribution
            APinc = ap_sum / max(len(u_rank), 1)

            scores.append(np.sqrt(max(LIN, 0.0) * max(APinc, 0.0)))

        return float(np.mean(scores))

    @staticmethod
    def compute_metrics(src_emb, tgt_emb, src_words, tgt_words, test_dict, method_name, plot_dir, is_sparse=False):
        src_map = {w: i for i, w in enumerate(src_words)}
        tgt_map = {w: i for i, w in enumerate(tgt_words)}
        
        valid_pairs = []
        for s, t in test_dict:
            if s in src_map and t in tgt_map:
                valid_pairs.append((src_map[s], tgt_map[t]))
        
        if not valid_pairs:
            return {"method": method_name}

        src_indices = [p[0] for p in valid_pairs]
        tgt_indices = [p[1] for p in valid_pairs]
        
        sub_src = src_emb[src_indices]
        #check if target and source are normalized
        norms = np.linalg.norm(sub_src, axis=1, keepdims=True)
        if not np.allclose(norms, 1.0):
            print("Source embeddings not normalized, normalizing now...")
            sub_src = sub_src / norms
        tgt_emb_norms = np.linalg.norm(tgt_emb, axis=1, keepdims=True)
        if not np.allclose(tgt_emb_norms, 1.0):
            print("Target embeddings not normalized, normalizing now...")   
            tgt_emb = tgt_emb / tgt_emb_norms

        # sim_matrix=np.dot(sub_src, tgt_emb.T)
        
        # 1. CSLS Calculation (on dense version or subset)
        # For huge sparse, this might need optimization, assuming fits in memory here
        
        csls_matrix = Evaluator.get_csls_sim(sub_src, tgt_emb)
        
        p1, p5 = 0, 0
        ap_sum = 0
        n = len(valid_pairs)
        print(f"Evaluating {n} valid pairs...")

        for i in range(n):
            target_true_index= tgt_indices[i]
            scores= csls_matrix[i]#get scores for query against all target words
            true_score=scores[target_true_index]
            rank=np.sum(scores>true_score)+1#rank is 1 + number of scores greater than true score
            if rank==1:p1+=1
            if rank<=5:p5+=1
            if i < 3 and rank > 1000:
                print(f"  Fail: Pair ({src_words[src_indices[i]]} -> {tgt_words[target_true_index]})")
                print(f"    True Score: {true_score:.4f}, Rank: {rank}")
                top_idx = np.argmax(scores)
                print(f"    Top 1 Prediction: {tgt_words[top_idx]} ({scores[top_idx]:.4f})")

            ap_sum+=1.0/rank
        sub_tgt = tgt_emb[tgt_indices]
        # 2. RSA
        src_dist = 1 - np.dot(sub_src, sub_src.T) 
        tgt_dist = 1 - np.dot(sub_tgt, sub_tgt.T)
        
        tri_idx = np.triu_indices(n, k=1)
        rsa_score = 0
        if len(tri_idx[0]) > 0:
            rsa_score = np.corrcoef(src_dist[tri_idx], tgt_dist[tri_idx])[0, 1]
            if np.isnan(rsa_score): rsa_score = 0
        
        # 3. balAPinc
        bal_score = 0
        if is_sparse:
            bal_score = Evaluator.bal_ap_inc(src_emb[src_indices], tgt_emb[tgt_indices])

        # Plot RSA
        plt.figure(figsize=(6, 6))
        plt.scatter(src_dist[tri_idx][::20], tgt_dist[tri_idx][::20], alpha=0.1, s=1)
        plt.title(f"{method_name} (RSA={rsa_score:.3f})")
        plt.savefig(os.path.join(plot_dir, f"rsa_{method_name}.png"))
        plt.close()

        return {
            "method": method_name,
            "P@1": p1 / n,
            "P@5": p5 / n,
            "MAP": ap_sum / n,
            "RSA": rsa_score,
            "balAPinc": bal_score if is_sparse else "N/A"
        }

class BiSparseSolver:
    def __init__(
        self,
        n_src,
        n_tgt,
        n_dim_in,
        n_atoms=100,
        lambda_e=0.5,
        lambda_x=10.0,
        device='cpu',
        seed=0,
    ):
        self.n_atoms = n_atoms
        self.lambda_e_max = float(lambda_e)   # treat as "max" for schedules
        self.lambda_e = float(lambda_e)       # current value (will be scheduled)
        self.lambda_x = float(lambda_x)
        self.device = device
        self.rng = torch.Generator(device='cpu').manual_seed(seed)

        common_D = torch.randn(n_atoms, n_dim_in, device=device)
        self.D_s = F.normalize(common_D, p=2, dim=1)
        self.D_t = self.D_s.clone()

        self.A_s = None
        self.A_t = None

    def _prox_nonneg_l1(self, x, alpha):
        # prox for lambda*||x||_1 with x>=0  => max(x-alpha, 0)  [web:64]
        return torch.relu(x - alpha)

    @torch.no_grad()
    def _diagnostics(self, src_idx, tgt_idx, tag=""):
        nnz_s = (self.A_s > 0).sum(dim=1).float().mean().item()
        nnz_t = (self.A_t > 0).sum(dim=1).float().mean().item()

        seed_diff = self.A_s[src_idx] - self.A_t[tgt_idx]
        seed_dist = torch.norm(seed_diff, dim=1).mean().item()

        m = src_idx.numel()
        rand_src = torch.randint(0, self.A_s.shape[0], (m,), device=self.device)
        rand_tgt = torch.randint(0, self.A_t.shape[0], (m,), device=self.device)
        rand_diff = self.A_s[rand_src] - self.A_t[rand_tgt]
        rand_dist = torch.norm(rand_diff, dim=1).mean().item()

        print(f"    [Diag{tag}] nnz_s={nnz_s:.2f} nnz_t={nnz_t:.2f} | "
              f"seed_L2={seed_dist:.4f} rand_L2={rand_dist:.4f}")

    def _update_A_ista(self, X_s, X_t, src_idx, tgt_idx, max_iter=5):
        """
        Prox-grad / ISTA:
          min_A  (1/2)||A_s D_s - X_s||^2 + lambda_e ||A_s||_1
               + (1/2)||A_t D_t - X_t||^2 + lambda_e ||A_t||_1
               + (lambda_x/2) * mean_seeds ||A_s[i] - A_t[j]||^2
        with A_s, A_t >= 0 handled by nonnegative L1 prox. [web:64]
        """
        N_s = X_s.shape[0]
        N_t = X_t.shape[0]
        M = src_idx.numel()

        DtD_s = self.D_s @ self.D_s.T
        DtD_t = self.D_t @ self.D_t.T
        XDt_s = X_s @ self.D_s.T
        XDt_t = X_t @ self.D_t.T

        # Larger step during warmup (lambda_e==0), smaller once shrinkage is active
        lr = 0.2 if self.lambda_e == 0.0 else 0.02

        for it in range(max_iter):
            grad_s = (self.A_s @ DtD_s - XDt_s) / N_s
            grad_t = (self.A_t @ DtD_t - XDt_t) / N_t

            diff = self.A_s[src_idx] - self.A_t[tgt_idx]  # [M, n_atoms]
            scale = self.lambda_x / max(M, 1)

            grad_c_s = torch.zeros_like(self.A_s)
            grad_c_t = torch.zeros_like(self.A_t)
            grad_c_s.index_add_(0, src_idx, scale * diff)
            grad_c_t.index_add_(0, tgt_idx, -scale * diff)

            self.A_s = self.A_s - lr * (grad_s + grad_c_s)
            self.A_t = self.A_t - lr * (grad_t + grad_c_t)

            if it == 0:
                print("    pre-prox A_s max/mean:",
                      self.A_s.max().item(), self.A_s.mean().item(),
                      "thr:", lr * self.lambda_e)

            self.A_s = self._prox_nonneg_l1(self.A_s, lr * self.lambda_e)
            self.A_t = self._prox_nonneg_l1(self.A_t, lr * self.lambda_e)

    def _update_D_block(self, X_s, X_t, lr=0.5):
        N_s = X_s.shape[0]
        N_t = X_t.shape[0]

        resid_s = (self.A_s @ self.D_s - X_s)
        resid_t = (self.A_t @ self.D_t - X_t)

        grad_D_s = (self.A_s.T @ resid_s) / N_s
        grad_D_t = (self.A_t.T @ resid_t) / N_t

        self.D_s -= lr * grad_D_s
        self.D_t -= lr * grad_D_t

        self.D_s = F.normalize(self.D_s, p=2, dim=1)
        self.D_t = F.normalize(self.D_t, p=2, dim=1)

    def fit(
        self,
        X_src,
        X_tgt,
        seed_pairs,
        epochs=200,
        ista_iters=5,              # keep small (5–10)
        warmup=5,
        ramp=80,
        target_nnz=12.0,
        tol=2.0,
        d_freeze_epochs=30,
    ):
        X_s = torch.tensor(X_src, dtype=torch.float32, device=self.device)
        X_t = torch.tensor(X_tgt, dtype=torch.float32, device=self.device)
        X_s = F.normalize(X_s, p=2, dim=1)
        X_t = F.normalize(X_t, p=2, dim=1)

        src_idx = torch.tensor([p[0] for p in seed_pairs], dtype=torch.long, device=self.device)
        tgt_idx = torch.tensor([p[1] for p in seed_pairs], dtype=torch.long, device=self.device)

        print("  [BiSparse] Initializing...")
        self.A_s = F.relu(X_s @ self.D_s.T) * 5.0
        self.A_t = F.relu(X_t @ self.D_t.T) * 5.0

        # start scheduled value (will change)
        self.lambda_e = 0.0

        print(f"  [BiSparse] Fit (lambda_e_max={self.lambda_e_max}, lambda_x={self.lambda_x}, atoms={self.n_atoms})...")

        for epoch in range(epochs):
            # --- adaptive lambda_e schedule (prevents ramping into all-zero) ---
            if epoch < warmup:
                self.lambda_e = 0.0
            else:
                # slow proposed ramp toward lambda_e_max
                t = min(1.0, (epoch - warmup) / max(ramp, 1))
                proposed = self.lambda_e_max * t

                with torch.no_grad():
                    nnz_s = (self.A_s > 0).sum(dim=1).float().mean().item()
                    nnz_t = (self.A_t > 0).sum(dim=1).float().mean().item()
                    nnz = 0.5 * (nnz_s + nnz_t)

                if nnz < (target_nnz - tol):
                    # already too sparse -> back off (reduce sparsity pressure)
                    self.lambda_e = 0.5 * proposed
                elif nnz > (target_nnz + tol):
                    # not sparse enough -> keep ramping
                    self.lambda_e = proposed
                else:
                    # in the sweet spot -> freeze at current value (don't keep increasing)
                    self.lambda_e = min(self.lambda_e, proposed) if epoch > warmup else proposed

            # --- updates ---
            self._update_A_ista(X_s, X_t, src_idx, tgt_idx, max_iter=ista_iters)

            # freeze D early to avoid destabilizing the codes
            if epoch >= d_freeze_epochs:
                self._update_D_block(X_s, X_t)

            # --- logging ---
            if epoch % 5 == 0:
                with torch.no_grad():
                    rec_s = torch.mean(torch.sum((self.A_s @ self.D_s - X_s) ** 2, dim=1)).item()
                    rec_t = torch.mean(torch.sum((self.A_t @ self.D_t - X_t) ** 2, dim=1)).item()
                    diff = self.A_s[src_idx] - self.A_t[tgt_idx]
                    c_loss = (self.lambda_x * 0.5 * torch.mean(torch.sum(diff ** 2, dim=1))).item()
                    l1_s = self.A_s.sum(dim=1).mean().item()
                    l1_t = self.A_t.sum(dim=1).mean().item()

                    print(f"    lambda_e={self.lambda_e:.4f} | L1mean_s={l1_s:.4f} L1mean_t={l1_t:.4f}")
                    print(f"    Epoch {epoch}: Rec_s={rec_s:.4f} Rec_t={rec_t:.4f} | C_loss={c_loss:.4f}")
                    self._diagnostics(src_idx, tgt_idx, tag=f"@{epoch}")

        return self.A_s.cpu().numpy(), self.A_t.cpu().numpy()


class Aligner:
    def __init__(self, output_dir):
        self.output_dir = output_dir

    def apply_procrustes(self, X, Y, src_idx, tgt_idx):
        """ Procrustes Analysis (PA)"""
        print(f"  [Procrustes] Aligning with {len(src_idx)} pairs.")
        A = X[src_idx]
        B = Y[tgt_idx]
        M = np.dot(A.T, B)
        U, S, Vt = scipy.linalg.svd(M)
        W = np.dot(U, Vt)
        W_=np.dot(Vt.T,U.T)
        if W.all()!=W_.all():
            print("  [Procrustes] Warning: SVD inconsistency detected.")

         # Diagnostics
        A_aligned = np.dot(A, W)
        train_sim = np.sum(A_aligned * B, axis=1)
        print(f"  [Procrustes] Mean Train Cosine Sim: {np.mean(train_sim):.4f}")
        return W
    def apply_gpa(self, X, Y, src_idx, tgt_idx, n_iter=50, tol=1e-6, seed="avg"):
        A=X[src_idx]
        B=Y[tgt_idx]
        d=A.shape[1]

        if seed=="avg":
            G=0.5*(A+B)
        elif seed=="src":
            G=A.copy()
        elif seed=="tgt":
            G=B.copy()
        else:
            raise ValueError("Invalid seed option for GPA.")
        T1=np.eye(d)
        T2=np.eye(d)

        for it in range(n_iter):
            M1=G.T@A
            U, _, Vt=scipy.linalg.svd(M1, full_matrices=False)
            T1=Vt.T@U.T

            M2=G.T@B
            U, _, Vt=scipy.linalg.svd(M2, full_matrices=False)
            T2=Vt.T@U.T

            G_new=0.5*(A@T1+B@T2)
            rel_change=np.linalg.norm(G_new-G)/(np.linalg.norm(G)+1e-12)
            G=G_new
            print(f"  [GPA] Iteration {it}: Rel Change={rel_change:.8f}")
            if it%5==0:
                W=T1@T2.T
                curr=np.mean(np.sum((A@W)*B, axis=1))
                print(f"[GPA] Intermediate Mean Train Cosine Sim: {curr:.4f}")
            if rel_change<tol:
                print(f"  [GPA] Converged at iteration {it}.")
                break
        W=T1@T2.T        
        A_to_B=A@W
        train_sim=np.sum(A_to_B*B, axis=1)
        print(f"  [GPA] Mean Train Cosine Sim: {np.mean(train_sim):.4f}")
        return W
    def apply_fgw(self, X, Y, src_words, tgt_words, seed_dict, alpha=0.1, mass=0.8, reg=0.002):
        """Fused Gromov-Wasserstein alignment."""
        n = min(len(X), 4000) 
        X_sub = X[:n]
        Y_sub = Y[:n]
        
        C1 = ot.dist(X_sub, X_sub)
        C2 = ot.dist(Y_sub, Y_sub)
        C1 /= (C1.max()+1e-9)
        C2 /= (C2.max()+1e-9)
        
        M = np.ones((n, n))*10
        src_map = {w: i for i, w in enumerate(src_words[:n])}
        tgt_map = {w: i for i, w in enumerate(tgt_words[:n])}
        
        seed_count = 0
        for s, t in seed_dict:
            if s in src_map and t in tgt_map:
                M[src_map[s], tgt_map[t]] = 0.0
                seed_count += 1
                
        print(f"  [FGW] Used {seed_count} seeds. Solving OT...")
        p, q = ot.unif(n)*mass, ot.unif(n)*mass
        try:
            T=ot.gromov.entropic_fused_gromov_wasserstein(M, C1, C2, p,q,loss_fun='square_loss',alpha=alpha,epsilon=reg, max_iter=1000,solver='ppg',verbose=True)
        except Exception as e:
            T=ot.gromov.fused_gromov_wasserstein(M,C1,C2,p,q,loss_fun='square_loss',alpha=alpha)
        np.save(os.path.join(self.output_dir, "fgw_transport_matrix.npy"), T)
        
        threshold = 1.0 / (n * 10)
        src_ind, tgt_ind = np.where(T > threshold)
        
        # Sort by confidence (weight in T)
        weights = T[src_ind, tgt_ind]
        sort_idx = np.argsort(-weights)
        
        # Keep top k strongest matches
        keep_k = min(len(sort_idx), 2500)
        final_src = src_ind[sort_idx[:keep_k]]
        final_tgt = tgt_ind[sort_idx[:keep_k]]
        
        print(f"  [Robust FGW] Found {len(final_src)} robust correspondences (discarded {n - len(final_src)} outliers).")
        
        return final_src, final_tgt

    def apply_bisparse(self,
    X_src,
    X_tgt,
    seed_pairs,
    n_atoms=100,
    lambda_e=0.5,
    lambda_x=10.0,
    epochs=60,
    ista_iters=30,
    device='cpu'
    ):
        solver = BiSparseSolver(
            len(X_src), len(X_tgt), X_src.shape[1],
            n_atoms=n_atoms, lambda_e=lambda_e, lambda_x=lambda_x,
            device=device
        )
        return solver.fit(X_src, X_tgt, seed_pairs, epochs=epochs, ista_iters=ista_iters)

def main():
    device = get_device()
    handler = DataHandler()
    aligner = Aligner(handler.matrices_dir)
    
    # 1. Load Data
    try:
        X_src, src_words, src_mat = handler.load_vectors("wiki.en.vec", n_max=50000)
        X_tgt, tgt_words, tgt_mat = handler.load_vectors("wiki.hi.vec", n_max=50000)
        gt_dict = handler.load_dictionary("en-hi.txt")

    except FileNotFoundError:
        print("Demo Mode: Generating random data")
        X_src = np.random.randn(1000, 300).astype(np.float32)
        X_tgt = np.random.randn(1000, 300).astype(np.float32)
        # Normalize
        X_src /= np.linalg.norm(X_src, axis=1, keepdims=True)
        X_tgt /= np.linalg.norm(X_tgt, axis=1, keepdims=True)
        src_mat, tgt_mat = X_src, X_tgt
        src_words = [f"w{i}" for i in range(1000)]
        tgt_words = [f"w{i}" for i in range(1000)]
        gt_dict = [(f"w{i}", f"w{i}") for i in range(500)]
    
    # Create lookup maps 
    src_map = {w: i for i, w in enumerate(src_words)}
    tgt_map = {w: i for i, w in enumerate(tgt_words)}
    
    print(f"\nRaw Dictionary Size: {len(gt_dict)}")
    
    valid_dict_pairs = []
    for s, t in gt_dict:
        # Check existence in both source and target vocab
        if s in src_map and t in tgt_map:
            valid_dict_pairs.append((s, t))
            
    print(f"Valid Dictionary Overlap: {len(valid_dict_pairs)} (These exist in both embedding spaces)")
    
    if len(valid_dict_pairs) == 0:
        raise ValueError("No dictionary words found in embeddings! Check casing or vocab size.")

    np.random.seed(42)
    np.random.shuffle(valid_dict_pairs)
    
    split_ratio = 0.8
    split_idx = int(len(valid_dict_pairs) * split_ratio)
    
    train_dict = valid_dict_pairs[:split_idx]
    test_dict = valid_dict_pairs[split_idx:]
    
    print(f"Train Size: {len(train_dict)} | Test Size: {len(test_dict)}")
    
    train_pairs_idx = []
    for s, t in train_dict:
        train_pairs_idx.append((src_map[s], tgt_map[t]))

    results = []


    # Baseline
    print("\n--- Baseline ---")
    handler.save_embedding("baseline_src", src_mat)
    res=Evaluator.compute_metrics(src_mat, tgt_mat, src_words, tgt_words, test_dict, "Baseline", handler.plots_dir)
    results.append(res)

    #PA
    print("\n--- PA ---")
    src_p = [p[0] for p in train_pairs_idx]
    tgt_p = [p[1] for p in train_pairs_idx]
    
    W_pa = aligner.apply_procrustes(src_mat, tgt_mat, src_p, tgt_p)
    X_src_pa = np.dot(src_mat, W_pa)
    
    handler.save_embedding("pa_src", X_src_pa)
    res = Evaluator.compute_metrics(X_src_pa, tgt_mat, src_words, tgt_words, test_dict, "PA", handler.plots_dir)
    results.append(res)

    #GPA
    W_gpa = aligner.apply_gpa(src_mat, tgt_mat, src_p, tgt_p, n_iter=100, tol=1e-6, seed="avg")
    X_src_gpa = np.dot(src_mat, W_gpa)
    handler.save_embedding("gpa_src", X_src_gpa)
    res = Evaluator.compute_metrics(X_src_gpa, tgt_mat, src_words, tgt_words, test_dict, "GPA", handler.plots_dir)
    results.append(res)

    # FGW
    print("\n--- FGW ---")
    # FGW finds new correspondence indices
    fgw_src_idx, fgw_tgt_idx = aligner.apply_fgw(src_mat, tgt_mat, src_words, tgt_words, train_dict,alpha=0.5,mass=0.9,reg=0.001)
    
    # We then use Procrustes on these found correspondences
    W_fgw = aligner.apply_procrustes(src_mat, tgt_mat, fgw_src_idx, fgw_tgt_idx)
    X_src_fgw = np.dot(src_mat, W_fgw)
    
    handler.save_embedding("fgw_src", X_src_fgw)
    res = Evaluator.compute_metrics(X_src_fgw, tgt_mat, src_words, tgt_words, test_dict, "FGW", handler.plots_dir)
    results.append(res)

    # --- 4. BiSparse (Eq 4) on Raw ---
    # print("\n--- BiSparse (Eq 4) ---")
    # Uses seed dictionary constraints to learn joint sparse space
    # A_s, A_t = aligner.apply_bisparse(X_src_gpa, tgt_mat, train_pairs_idx,n_atoms=100, lambda_e=0.5, lambda_x=10.0,epochs=80, ista_iters=40, device=device)
    # A_s, A_t = aligner.apply_bisparse(
    #     X_src_gpa, tgt_mat, train_pairs_idx,
    #     n_atoms=100, lambda_e=0.5, lambda_x=10.0,
    #     epochs=200, ista_iters=5, device=device
    # )
    # handler.save_embedding("bisparse_src", A_s)
    # handler.save_embedding("bisparse_tgt", A_t)
    # res = Evaluator.compute_metrics(A_s, A_t, src_words, tgt_words, test_dict, "BiSparse", handler.plots_dir, is_sparse=True)
    # results.append(res)
    
    # # --- 5. FGW + BiSparse ---
    # print("\n--- FGW + BiSparse ---")
    # # FGW gives us synthetic pairs (fgw_src_idx, fgw_tgt_idx) which we add to training seeds
    # aug_train_pairs = list(zip(fgw_src_idx, fgw_tgt_idx))
    # # Combine with original seeds (optional, but robust)
    # full_train_pairs = list(set(train_pairs_idx + aug_train_pairs))
    
    # print(f"  [FGW+BiSparse] Training with {len(full_train_pairs)} augmented pairs.")
    # A_s_f, A_t_f = aligner.apply_bisparse(src_mat, tgt_mat, full_train_pairs, device=device)
    
    # handler.save_embedding("fgw_bisparse_src", A_s_f)
    # handler.save_embedding("fgw_bisparse_tgt", A_t_f)
    # res = Evaluator.compute_metrics(A_s_f, A_t_f, src_words, tgt_words, test_dict, "FGW+BiSparse", handler.plots_dir, is_sparse=True)
    # results.append(res)

    # Save
    df = pd.DataFrame(results)
    df.to_csv("metrics_summary50k_hindi_bisparse.csv", index=False)
    print("\nEvaluation Complete. Summary saved to metrics_summary50k_hindi_bisparse.csv")
    print(df)

if __name__ == "__main__":
    main()

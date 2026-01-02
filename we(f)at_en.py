import os
import re
import argparse
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import requests
from datasets import load_dataset

token_split=re.compile(r"[^\w]+", flags=re.UNICODE)

def pearsonr(x:np.ndarray, y:np.ndarray) ->float:
    x=x.astype(np.float64)
    y=y.astype(np.float64)
    x0=x - x.mean()
    y0=y - y.mean()
    return float(np.dot(x0,y0)/((np.linalg.norm(x0)*np.linalg.norm(y0))+1e-12))

def rankdata(a:np.ndarray) -> np.ndarray:
    order=np.argsort(a)
    ranks=np.empty_like(order,dtype=np.float64)
    ranks[order]=np.arange(1,len(a)+1,dtype=np.float64)
    sorted_a=a[order]
    i=0
    while i<len(a):
        j=i
        while j+1<len(a) and sorted_a[j+1]==sorted_a[i]:
            j+=1
        if j>i:
            avg=ranks[order[i:j+1]].mean()
            ranks[order[i:j+1]]=avg
        i=j+1
    return ranks
def spearmanr(x:np.ndarray, y:np.ndarray) ->float:
    return pearsonr(rankdata(x), rankdata(y))    


def load_weat_tests(cache_dir: str):
    words = load_dataset("fairnlp/weat", data_files=["words.parquet"], split="train", cache_dir=cache_dir)
    assoc = load_dataset("fairnlp/weat", data_files=["associations_weat.parquet"], split="train", cache_dir=cache_dir)

    print("words cols:", words.column_names)
    print("assoc cols:", assoc.column_names)
    print("example words row:", words[0])

    class2words = {}

    cols = set(words.column_names)
    if ("words" in cols) and ("id" in cols):
        for ex in words:
            cname = str(ex["id"])
            wlist = ex["words"]
            class2words[cname] = [str(w).lower() for w in wlist]
    elif ("classes" in cols) and (("word" in cols) or ("id" in cols)):
        word_col = "word" if "word" in cols else "id"
        for ex in words:
            w = str(ex[word_col]).lower()
            c = ex["classes"]
            if isinstance(c, list):
                for ci in c:
                    class2words.setdefault(str(ci), []).append(w)
            else:
                class2words.setdefault(str(c), []).append(w)

    else:
        raise ValueError(f"Unexpected words.parquet schema: {words.column_names}")

    return class2words, list(assoc)


def materialize_weat_test(class2words: Dict[str, List[str]], row: dict)-> Tuple[List[str],List[str],List[str],List[str],List[str]]:
    test_id=str(row.get("id",row.get("test","unknown")))
    X=[w.lower() for w in class2words.get(str(row["X"]),[])]
    Y=[w.lower() for w in class2words.get(str(row["Y"]),[])]
    A=[w.lower() for w in class2words.get(str(row["A"]),[])]
    B=[w.lower() for w in class2words.get(str(row["B"]),[])]
    return test_id,X,Y,A,B

def load_bls_pct_women(max_rows: int) -> pd.DataFrame:
    import pandas as pd
    import requests
    from io import BytesIO

    url = "https://www.bls.gov/cps/cpsaat11.htm"

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0 Safari/537.36"
        ),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
    }

    resp = requests.get(url, headers=headers, timeout=30)
    resp.raise_for_status()

    tables = pd.read_html(BytesIO(resp.content), flavor="lxml")  # requires lxml installed

    df = max(tables, key=lambda t: t.shape[0])
    # If read_html produced MultiIndex columns, flatten them into plain strings
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = ['_'.join([str(x) for x in col if str(x) != "nan"]).strip()
                    for col in df.columns.to_flat_index()]
    else:
        df.columns = [str(c) for c in df.columns]
    print(df.columns)
    
    cols = [str(c) for c in df.columns]
    occ_col = next((c for c in cols if "occupation" in c.lower()), cols[0])
    women_col = next((c for c in cols if "women" in c.lower() or "female" in c.lower()), None)
    if women_col is None:
        raise ValueError(f"Could not find women/female column in: {cols}")

    out = df[[occ_col, women_col]].copy()
    out.columns = ["occupation", "pct_women"]
    out["pct_women"] = pd.to_numeric(out["pct_women"], errors="coerce")
    out = out.dropna(subset=["pct_women"])
    out["occupation"] = out["occupation"].astype(str).str.strip()
    out = out[(out["pct_women"] >= 0.0) & (out["pct_women"] <= 100.0)]

    return out.head(max_rows).reset_index(drop=True)

def load_fasttext_vec_selective(vec_path: str, needed: set) -> Dict[str, np.ndarray]:
    """
    Loads only vectors for words in `needed` from a fastText .vec file.

    Robust parsing:
      - read header: "<vocab> <dim>" (fastText text format)
      - for each line: last `dim` entries are the vector, everything before is the token
        (handles tokens that contain spaces)
    """
    import os
    import numpy as np

    if not os.path.exists(vec_path):
        raise FileNotFoundError(vec_path)

    words = []
    vecs = []

    with open(vec_path, "r", encoding="utf-8", errors="ignore") as f:
        header = f.readline().strip().split()
        if len(header) != 2 or (not header[0].isdigit()) or (not header[1].isdigit()):
            raise ValueError("Expected fastText .vec header: '<vocab_size> <dim>' on the first line.")
        dim = int(header[1])

        for line in f:
            parts = line.rstrip().split()
            if len(parts) <= dim:
                continue 

            token = " ".join(parts[:-dim]).lower()
            if token not in needed:
                continue
            try:
                v = np.asarray(parts[-dim:], dtype=np.float32)
            except ValueError:
                continue

            words.append(token)
            vecs.append(v)

    if not vecs:
        raise ValueError("No needed words found in the embedding file. Check your `needed` set and casing.")

    mat = np.vstack(vecs)
    mat = mat - mat.mean(axis=0, keepdims=True)
    mat = mat / (np.linalg.norm(mat, axis=1, keepdims=True) + 1e-9)

    return {w: mat[i] for i, w in enumerate(words)}

def single_token_vector(emb: Dict[str, np.ndarray], text: str) -> Optional[np.ndarray]:
    """
    Return the embedding for `text` only if it tokenizes to a *single* token.
    Otherwise return None (drop multi-word / multi-token phrases).
    """
    toks = [t for t in token_split.split(text.lower().strip()) if t]
    if len(toks) != 1:
        return None
    v = emb.get(toks[0], None)
    if v is None:
        return None
    return v / (np.linalg.norm(v) + 1e-9)


#weat
def assoc(w: np.ndarray, A: np.ndarray, B: np.ndarray) -> float:
    """Association of word w with attribute sets A and B"""
    return float(np.mean(A@w) - np.mean(B@w))

def weat_stat(X: np.ndarray, Y: np.ndarray, A: np.ndarray, B: np.ndarray) -> float:
    """WEAT test statistic for target sets X and Y with attribute sets A and B"""
    return float(np.sum([assoc(x, A, B) for x in X]) - np.sum([assoc(y, A, B) for y in Y]))

def weat_effect_size(X: np.ndarray, Y: np.ndarray, A: np.ndarray, B: np.ndarray) -> float:
    sx = np.array([assoc(x, A, B) for x in X], dtype=np.float64)
    sy = np.array([assoc(y, A, B) for y in Y], dtype=np.float64)
    sall = np.concatenate([sx, sy], axis=0)
    return float((sx.mean() - sy.mean()) / (sall.std(ddof=1) + 1e-12))

def weat_pvalue_mc(X: np.ndarray, Y: np.ndarray, A: np.ndarray, B: np.ndarray, n_samples: int, seed: int) -> float:
    rng = np.random.default_rng(seed)
    s_obs = weat_stat(X, Y, A, B)
    T = np.concatenate([X, Y], axis=0)
    nx = X.shape[0]
    nt = T.shape[0]
    cnt = 0
    for _ in range(n_samples):
        idx = rng.permutation(nt)
        Xi = T[idx[:nx]]
        Yi = T[idx[nx:]]
        if weat_stat(Xi, Yi, A, B) >= s_obs:
            cnt += 1
    return float((cnt + 1) / (n_samples + 1))

def to_matrix(emb: Dict[str, np.ndarray], ws: List[str]) -> np.ndarray:
    vs = [emb[w] for w in ws if w in emb]
    if not vs:
        return np.zeros((0, 1), dtype=np.float32)
    return np.stack(vs, axis=0).astype(np.float32)


#wefat-occupation gender

male = ["he", "him", "his", "man", "male", "boy", "father", "son"]
female = ["she", "her", "hers", "woman", "female", "girl", "mother", "daughter"]

def gender_assoc_score(emb: Dict[str, np.ndarray], item_phrase: str) -> Optional[float]:
    v = single_token_vector(emb, item_phrase)
    if v is None:
        return None
    A = to_matrix(emb, male)
    B = to_matrix(emb, female)
    if A.shape[0] < 2 or B.shape[0] < 2:
        return None
    return float(np.mean(A @ v) - np.mean(B @ v))



def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--vec_path", type=str,default='wiki.en.vec', help="Path to wiki.en.vec (fastText .vec text file)")
    ap.add_argument("--out_dir", type=str,default='./bias/data2', help="Directory to store CSV outputs")
    ap.add_argument("--weat_ids",type=str,default="all",help=(
        "Comma-separated WEAT test IDs from fairnlp/weat "
        "(e.g., male_female_career_family,flowers_insects_pleasant_unpleasant) "
        "or 'all' to run all available WEAT tests"))
    ap.add_argument("--perm_samples", type=int, default=50000, help="Monte Carlo permutation samples for p-value")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max_bls_rows", type=int, default=500, help="Max occupations to use from BLS table")
    args = ap.parse_args()

    ALL_WEAT_IDS = [
    "flowers_insects_pleasant_unpleasant",
    "europeanamerican_africanamerican_pleasant_unpleasant",
    "male_female_career_family",
    "musicalinstruments_weapons_pleasant_unpleasant",
    "science_arts_male_female",
    "math_arts_male_female"
    ]
    if args.weat_ids.lower() == "all":
        weat_ids = ALL_WEAT_IDS
    else:
        weat_ids = [w.strip() for w in args.weat_ids.split(",") if w.strip()]

        


    os.makedirs(args.out_dir, exist_ok=True)

    class2words, assoc_rows = load_weat_tests(cache_dir=os.path.join(args.out_dir, "hf_cache"))
    wanted = set(weat_ids)
    assoc_rows = [r for r in assoc_rows if str(r.get("id", r.get("test", ""))) in wanted ]

    if not assoc_rows:
        raise ValueError("No WEAT tests matched --weat_ids. Check IDs in the fairnlp/weat dataset viewer.")

    bls=load_bls_pct_women(max_rows=args.max_bls_rows)

    needed=set()
    for r in assoc_rows:
        _,X,Y,A,B=materialize_weat_test(class2words,r)
        print(_, len(X), len(Y), len(A), len(B))
        print(_, X, Y, A, B)
        needed |= set(X) | set(Y) | set(A) | set(B)
    for occ in bls["occupation"].tolist():
        toks = [t for t in token_split.split(occ.lower().strip()) if t]
        if len(toks) == 1:
            needed.add(toks[0])
    needed |= set(male) | set(female)

    emb = load_fasttext_vec_selective(args.vec_path, needed)

    weat_out=[]
    for r in assoc_rows:
        test_id, Xw, Yw, Aw, Bw = materialize_weat_test(class2words, r)
        X = to_matrix(emb, Xw); Y = to_matrix(emb, Yw); A = to_matrix(emb, Aw); B = to_matrix(emb, Bw)

        if min(X.shape[0], Y.shape[0], A.shape[0], B.shape[0]) < 2:
            weat_out.append({
                "kind": "WEAT",
                "name": test_id,
                "effect_size": np.nan,
                "p_value": np.nan,
                "n_X": X.shape[0], "n_Y": Y.shape[0], "n_A": A.shape[0], "n_B": B.shape[0],
            })
            continue

        eff = weat_effect_size(X, Y, A, B)
        p = weat_pvalue_mc(X, Y, A, B, n_samples=args.perm_samples, seed=args.seed)
        weat_out.append({
            "kind": "WEAT",
            "name": test_id,
            "effect_size": eff,
            "p_value": p,
            "n_X": X.shape[0], "n_Y": Y.shape[0], "n_A": A.shape[0], "n_B": B.shape[0],
        })

    items = []
    labels = []
    scores = []
    for _, row in bls.iterrows():
        occ = str(row["occupation"])
        pct_women = float(row["pct_women"])
        s = gender_assoc_score(emb, occ)
        if s is None:
            continue
        items.append(occ)
        labels.append(pct_women)
        scores.append(s)

    labels = np.asarray(labels, dtype=np.float64)
    scores = np.asarray(scores, dtype=np.float64)

    if len(items) >= 5:
        pr = pearsonr(scores, labels)
        sr = spearmanr(scores, labels)
    else:
        pr = np.nan
        sr = np.nan

    wefat_row = {
    "kind": "WEFAT",
    "name": "BLS_pct_women_vs_gender_assoc",
    "pearson_r": pr,
    "spearman_r": sr,
    "n_items": len(items),
}


    summary = pd.DataFrame(weat_out + [wefat_row])
    summary_path = os.path.join(args.out_dir, "summary.csv")
    summary.to_csv(summary_path, index=False)

    per_occ = pd.DataFrame({"occupation": items, "pct_women": labels, "gender_assoc_score": scores})
    per_occ_path = os.path.join(args.out_dir, "wefat_per_occupation.csv")
    per_occ.to_csv(per_occ_path, index=False)

    print(f"Wrote: {summary_path}")
    print(f"Wrote: {per_occ_path}")

if __name__=="__main__":
    main()    


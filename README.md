# Word Embeddings 
The repository has five objectives
1. Frequency based embeddings 
2. Evaluating pretrained embeddings 
3. Aligning monolingual embeddings
4. Evaluating the bias in pretrained static embeddings using WEAT and WEFAT
5. Evaluating gender bias in contextual embeddings in encoder only and decoder only models

### Notes:
- While the evaluation datasets are present in the repository, the word embeddings (pretrained and trained) are not due to file size limitations.
- Please download Fasttext wiki embeddings (wiki.en.vec) and (wiki.hi.vec), (cc.fr.300.vec) for evaluation and alignment. Also download the training corpus eng-wikipedia_2016_1M from https://wortschatz.uni-leipzig.de/en/download/eng
- All code files support sparse matrices from scipy, this makes computation much faster and all experiments possible on cpu as well.
- We truncate the vocabulary from training corpus to 30k from around 100K to avoid super sparse and meaningless rows in co-occurrence matrix.
- Even for the wiki embeddings we often restricted to 30k vocab to make results comparable for analysis with trained embeddings. For alignment, we restricted to 50k.

## 1. Frequency based embeddings 
To calculate co-occurrence matrices and PPMI, evaluation on Wordsim 353, differential drift: download the text corpus and run:
```bash
 python cooccurrence_matrix/co-occ_matrix.py
 ```

(Uncomment the commented part for the initial results as in the report)

To compute the pruned svd,ppmi, vocab and evalute on differential_drift, wordsim run 
```bash 
python cooccurrence_matrix/svd_.py
```

To evaluate on these on Google analogy test, Outliers test, WordSimilarity, run
```bash
python cooccurrence_matrix/evaluate_metrics.py
```

## 2. Evaluating pretrained embeddings 
To evaluate pretrained embeddings on same metrics, run
```bash
python cooccurrence_matrix/evaluate_pretrained.py 
```
this currently only supports fasttest wiki embeddings

Evaluates on
Google Analogy test set
Outliers test set
WordSimilarity 353

To export pretrained/ trained embeddings to tensorflow projector compatible files

run
```bash
python cooccurrence_matrix/export_to_projector.py 
```

## 3. Aligning monolingual embeddings
run 
```bash
 python alignment.py
 ```

Currently has :
1. Procrustes analysis
2. Generalized proscustes analysis
3. Fused gromov Wasserstein
4. Bisparse alignment (broken- does not converge)

Evaluates using :
- P@1
- P@k
- MAP
- RSA
- balAPinc (for bisparse)

## 4. Evaluating the bias in pretrained static embeddings using WEAT and WEFAT
Run 
```bash
python we(f)at_en.py
```

For hindi run
```bash
python weat-hi.py
```


## 5. Evaluating gender bias in contextual embeddings in encoder only and decoder only models

Suggested to run these on colab/ kaggle on gpu as loading the models could be heavy

```gender_bias_encoder.ipynb```
```gender-bias-gemma2-2b-it.ipynb```

For encoder, please do restart the session after pip uninstall numpy , installing the correct version (as in the notebook) to avoid dependency issues with transformer-lens HookedEncoder


For set up

### Option 1: Pip (Recommended for most users)
```bash
git clone https://github.com/ilatims-b/word_embeddings.git
cd word_embeddings.git
python -m venv alignvec
source alignvec/bin/activate  # On Windows: alignvec\Scripts\activate
pip install -r requirements.txt
```
### Option 2:Conda (Full reproducibility, includes MPS/CUDA support)
```bash
conda env create -f environment.yml
conda activate alignvec
```




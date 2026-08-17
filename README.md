# Zero-Target Cross-Corpus Speech Emotion Recognition Across Three Acted Corpora

This repository contains the notebook pipeline used for an exploratory zero-target cross-corpus SER analysis across EmoDB 2.0, RAVDESS, and RESD. The study does not claim to causally disentangle language shift and dialogue-style shift; instead, it evaluates model behavior under strict cross-corpus conditions where language, corpus identity, recording conditions, label procedures, emotion distribution, and speaking style vary jointly.

---

## 1. Datasets

The experiments use three emotional speech corpora mapped to six shared emotion classes.

| Corpus | Language | Speaking style | Classes used |
|---|---|---|---|
| EmoDB 2.0 | German | acted monologue | angry, disgust, fear, happy, neutral, sad |
| RAVDESS | English | acted monologue | angry, disgust, fear, happy, neutral, sad |
| RESD | Russian | acted dialogue | angry, disgust, fear, happy, neutral, sad |

Labels not shared by all corpora are excluded, for example boredom in EmoDB 2.0, calm and surprised in RAVDESS, and enthusiasm in RESD.

The unified label set is:

```text
angry, disgust, fear, happy, neutral, sad
```

---

## 2. Project Directory

The notebooks assume the following base directory in Google Drive:

```python
BASE_PROJECT = Path("/content/drive/MyDrive/New Jurnal Cross")
```

Expected high-level structure:

```text
New Jurnal Cross/
├── data/                              # raw dataset folders, if used by preprocessing notebooks
├── processed_intra_csv/               # metadata and split CSV files
├── processed_intra_features_hc_noaug/ # utterance-level handcrafted features
├── processed_intra_features_e2v/      # utterance-level emotion2vec features
├── processed_intra_sequence_features/ # sequence-level emotion2vec and handcrafted features
├── results_intra_handcrafted_svm/
├── results_intra_handcrafted_mlp/
├── results_intra_emotion2vec_mlp/
├── results_intra_concat_fusion_mlp/
├── results_intra_cross_attention_fusion/
├── results_intra_sequence_cross_attention/
├── results_lodo_utterance_fusion/
├── results_lodo_sequence_cross_attention/
└── figures/
```

---

## 3. Notebook Execution Order

Run the notebooks in this order.

| Step | Notebook | Purpose | Main output |
|---:|---|---|---|
| 0 | `Metadata.ipynb` | Build metadata and speaker-independent splits | `processed_intra_csv/` |
| 1 | `02_feature_extraction_handcrafted_intra.ipynb` | Extract 548-dimensional utterance-level handcrafted features | `processed_intra_features_hc_noaug/` |
| 2 | `03_train_handcrafted_intra.ipynb` | Train Handcrafted SVM-RBF intra-corpus baseline | `results_intra_handcrafted_svm/` |
| 3 | `04_train_handcrafted_mlp_intra.ipynb` | Train Handcrafted MLP intra-corpus baseline | `results_intra_handcrafted_mlp/` |
| 4 | `05_extract_emotion2vec_intra.ipynb` | Extract 768-dimensional utterance-level emotion2vec embeddings | `processed_intra_features_e2v/` |
| 5 | `06_train_emotion2vec_intra.ipynb` | Train emotion2vec MLP intra-corpus model | `results_intra_emotion2vec_mlp/` |
| 6 | `07_train_concat_fusion_intra.ipynb` | Train utterance-level concat fusion MLP intra-corpus model | `results_intra_concat_fusion_mlp/` |
| 7 | `08_train_cross_attention_fusion_intra.ipynb` | Train modality-level cross-attention fusion model | `results_intra_cross_attention_fusion/` |
| 8 | `08b_extract_sequence_features_intra.ipynb` | Extract sequence-level features | `processed_intra_sequence_features/` |
| 9 | `09_train_sequence_cross_attention_intra.ipynb` | Train sequence-level cross-attention intra-corpus model | `results_intra_sequence_cross_attention/` |
| 10 | `10_train_lodo_utterance_fusion.ipynb` | Train LODO emotion2vec MLP and Concat Fusion MLP | `results_lodo_utterance_fusion/` |
| 11 | `11_train_lodo_sequence_cross_attention.ipynb` | Train LODO Sequence-level Cross-Attention | `results_lodo_sequence_cross_attention/` |

---

## 4. Preprocessing

All audio files are standardized before feature extraction:

```text
mono
16 kHz sampling rate
fixed duration = 4 seconds
```

Longer utterances are center-cropped, while shorter utterances are reflect-padded. This produces a fixed waveform length of:

```text
4 seconds × 16,000 Hz = 64,000 samples
```

No audio augmentation is used in the final experiments.

Feature scaling is fitted only on the training data for each protocol. In the LODO setting, the held-out target corpus is never used for training, validation, early stopping, model selection, or feature scaling.

---

## 5. Feature Representations

### 5.1 Utterance-level handcrafted features

The handcrafted acoustic-prosodic vector has **548 dimensions** and is used by:

- Handcrafted SVM-RBF
- Handcrafted MLP
- Concat Fusion MLP

The 548-dimensional vector includes:

| Feature group | Dimension |
|---|---:|
| MFCC mean + std | 80 |
| Delta MFCC mean + std | 80 |
| Delta-delta MFCC mean + std | 80 |
| Chroma mean + std | 24 |
| Log-Mel statistics | 256 |
| Spectral contrast mean + std | 14 |
| Spectral rolloff mean + std | 2 |
| Spectral bandwidth mean + std | 2 |
| Prosodic block: F0, jitter approximation, RMS, ZCR | 10 |
| **Total** | **548** |

### 5.2 Utterance-level emotion2vec features

The emotion2vec feature extractor produces one **768-dimensional embedding** per utterance:

```text
1 utterance → 768-dimensional emotion2vec embedding
```

This representation is used by:

- emotion2vec MLP
- Concat Fusion MLP

### 5.3 Sequence-level features

For sequence-level cross-attention, each utterance is represented as a fixed-length temporal sequence:

```text
T = 200 frames
```

The two sequence streams are:

| Stream | Shape |
|---|---:|
| emotion2vec sequence | `T × 768` |
| handcrafted sequence | `T × 43` |

The frame-level handcrafted vector contains:

```text
40 MFCC coefficients + RMS + ZCR + F0 = 43 dimensions per frame
```

The 548-dimensional handcrafted vector and the `T × 43` handcrafted sequence are intentionally different. The 548-dimensional vector summarizes the whole utterance using statistical functionals, while the `T × 43` representation preserves frame-level temporal structure for attention-based fusion.

---

## 6. Experimental Protocols

### 6.1 Intra-corpus evaluation

Each corpus is split into train, validation, and test sets using a speaker-independent protocol.

Target split ratio:

```text
train / validation / test = 70 / 15 / 15
```

The final utterance ratio may vary because speakers are not allowed to appear in more than one split, and each split is constrained to contain all six emotion classes.

### 6.2 LODO cross-corpus evaluation

LODO means **Leave-One-Dataset-Out**. In each fold, two datasets are used as source corpora and the remaining dataset is held out entirely for testing.

| Held-out test corpus | Source corpora | Shift condition |
|---|---|---|
| EmoDB 2.0 | RAVDESS + RESD | English/Russian → German; monologue/dialogue → monologue |
| RAVDESS | EmoDB 2.0 + RESD | German/Russian → English; monologue/dialogue → monologue |
| RESD | EmoDB 2.0 + RAVDESS | German/English → Russian; monologue → dialogue |

The held-out corpus is not used for training, validation, early stopping, model selection, or feature scaling.

---

## 7. Models

Five models are evaluated.

| Model | Input | Purpose |
|---|---|---|
| Handcrafted SVM-RBF | 548-dimensional handcrafted vector | classical handcrafted baseline |
| Handcrafted MLP | 548-dimensional handcrafted vector | neural handcrafted baseline |
| emotion2vec MLP | 768-dimensional emotion2vec embedding | pretrained representation baseline |
| Concat Fusion MLP | 768 emotion2vec + 548 handcrafted = 1316-dimensional vector | naive utterance-level fusion |
| Sequence-level Cross-Attention | `T × 768` emotion2vec + `T × 43` handcrafted | selective temporal fusion |

---

## 8. Model Hyperparameters

### 8.1 SVM-RBF

The Handcrafted SVM-RBF baseline uses:

```python
SVC(
    kernel="rbf",
    C=10.0,
    gamma="scale",
    class_weight="balanced",
    probability=True,
    random_state=seed
)
```

### 8.2 Handcrafted MLP

```text
Input: 548
Architecture: 548 → 256 → 128 → 6
Normalization: BatchNorm1d
Activation: ReLU
Dropout: 0.30
```

### 8.3 emotion2vec MLP

```text
Input: 768
Architecture: 768 → 256 → 128 → 6
Normalization: LayerNorm
Activation: ReLU
Dropout: 0.30
```

### 8.4 Concat Fusion MLP

```text
Input: 1316
Architecture: 1316 → 512 → 256 → 128 → 6
Normalization: LayerNorm
Activation: ReLU
Dropout: 0.35
```

### 8.5 Sequence-level Cross-Attention

```text
Input emotion2vec: T × 768
Input handcrafted: T × 43
T: 200 frames
d_model: 128
attention heads: 4
dropout: 0.30
cross-attention blocks: 1 bidirectional block
positional encoding: none
```

Projection layers:

```text
Linear(input_dim → 128)
LayerNorm(128)
ReLU
Dropout(0.30)
```

Bidirectional cross-attention:

```text
e2v_to_hc: Q = emotion2vec, K/V = handcrafted
hc_to_e2v: Q = handcrafted, K/V = emotion2vec
```

Pooling:

```text
attentive pooling
masked mean pooling
masked max pooling
```

Fusion vector:

```text
[e_att_pool, h_att_pool, e_mean, h_mean, e_max, h_max, |e-h|, e*h]
```

Since each component is 128-dimensional:

```text
8 × 128 = 1024-dimensional fused vector
```

Classifier:

```text
1024 → 512 → 256 → 6
LayerNorm + ReLU + Dropout(0.30)
```

---

## 9. Training Hyperparameters

All neural models use:

```text
optimizer: AdamW
loss: class-weighted CrossEntropyLoss
scheduler: ReduceLROnPlateau
scheduler factor: 0.5
scheduler patience: 5
monitor: validation Macro-F1
gradient clipping: max_norm = 5.0
random seeds: 42, 123, 2024
```

| Model group | Batch size | Learning rate | Max epochs | Early stopping patience |
|---|---:|---:|---:|---:|
| Handcrafted MLP | 32 | 1e-3 | 150 | 20 |
| emotion2vec MLP | 32 | 1e-3 | 150 | 20 |
| Concat Fusion MLP | 32 | 8e-4 | 150 | 20 |
| Sequence-level Cross-Attention intra-corpus | 16 | 3e-4 | 120 | 15 |
| Sequence-level Cross-Attention LODO | 8 | 3e-4 | 120 | 15 |

For LODO utterance-level experiments:

| Model | Batch size | Learning rate | Max epochs | Early stopping patience |
|---|---:|---:|---:|---:|
| emotion2vec MLP | 32 | 1e-3 | 150 | 20 |
| Concat Fusion MLP | 32 | 8e-4 | 150 | 20 |

---

## 10. Evaluation Metrics

The reported metrics are:

```text
Accuracy
Unweighted Average Recall (UAR)
Macro-F1
Weighted-F1
```

Macro-F1 and UAR are emphasized because class distributions are not perfectly balanced across corpora.

All neural results are reported as:

```text
mean ± standard deviation over three seeds
```

---

## 11. Main Outputs

The notebooks save per-run and summary artifacts such as:

```text
*_metrics.csv
*_classification_report.csv
*_confusion_matrix.csv
*_predictions.csv
*_paper_table*.csv
*_per_class*.csv
model checkpoints
trained SVM pickle files
```

Important summary files include:

```text
results_intra_concat_fusion_mlp/compare_all_intra_test.csv
results_intra_sequence_cross_attention/compare_all_intra_test.csv
results_lodo_utterance_fusion/lodo_paper_table_test.csv
results_lodo_utterance_fusion/lodo_per_class_summary_mean_std.csv
results_lodo_sequence_cross_attention/compare_lodo_all_models_paper_table.csv
results_lodo_sequence_cross_attention/lodo_sequence_xattn_per_class_summary_mean_std.csv
```

---

## 12. Main Experimental Findings

The main findings are:

1. **emotion2vec MLP is the strongest average LODO baseline.**
   - Average LODO Macro-F1: **79.02%**

2. **Naive concat fusion degrades cross-corpus performance.**
   - Average LODO Macro-F1 decreases to **73.63%**

3. **Sequence-level cross-attention improves over concat fusion.**
   - Average LODO Macro-F1 increases to **77.90%**
   - It consistently outperforms concat fusion in all LODO folds.

4. **RESD is the hardest held-out corpus.**
   - This suggests that language shift combined with monologue-to-dialogue shift is more difficult than language shift between acted monologue corpora.

---

## 13. Figures

The main figures used in the paper are:

| Figure | Description |
|---|---|
| Figure 1 | Fusion architectures: Concat Fusion MLP and Sequence-level Cross-Attention |
| Figure 2 | LODO Macro-F1 across held-out corpora with seed-level error bars |
| Figure 3 | Per-class LODO F1 heatmap across held-out corpora and models |
| Figure 4 | Per-class precision and recall for the RESD held-out fold |

---

## 14. Reproducibility Notes

- Use the same random seeds for all reported runs:

```python
SEEDS = [42, 123, 2024]
```

- Fit scalers only on training data.
- In LODO, never use the held-out target corpus for training, validation, model selection, or feature scaling.
- Use the same six-class label mapping for all corpora.
- Report mean and standard deviation over the three seeds.

---

## 15. Citation / Paper Context

This code supports the following manuscript:

```text
An Empirical Analysis of Language and Dialogue-Style Shifts in Cross-Corpus Speech Emotion Recognition
```

The study is framed as an analytical cross-corpus SER study, not as a state-of-the-art leaderboard claim.

---

## 16. Software Dependencies

The notebooks use Python and common speech/audio and machine learning libraries:

```text
numpy
pandas
scikit-learn
librosa
torch
matplotlib
seaborn
tqdm
funasr
```

Install example:

```bash
pip install numpy pandas scikit-learn librosa torch matplotlib seaborn tqdm funasr
```

If running on Google Colab, mount Google Drive first:

```python
from google.colab import drive
drive.mount('/content/drive')
```

---

## 17. Notes

- The project assumes that datasets are available locally or in Google Drive.
- Dataset licenses and usage conditions should be followed separately.
- RESD is accessed through the Aniemore project.
- The final experiments do not use audio augmentation.
- The LODO results are strict zero-target results.

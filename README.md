# Zero-Target Cross-Corpus Speech Emotion Recognition Across Three Acted Corpora

This repository contains the experimental notebook pipeline, split manifests, fold-level results, and figure-generation artifacts for the manuscript:

> **Zero-Target Cross-Corpus Speech Emotion Recognition Across Three Acted Corpora**

The study is an **exploratory zero-target cross-corpus speech emotion recognition (SER) analysis** using three acted emotional speech corpora: **EmoDB 2.0**, **RAVDESS**, and **RESD**. The study does **not** claim to causally disentangle language shift and dialogue-style shift. Instead, it evaluates how SER models behave under strict cross-corpus conditions where **language, corpus identity, recording conditions, label procedures, emotion distribution, and speaking style vary jointly**.

The repository is intended to support reproducibility and reviewer verification of the zero-target LODO and directional transfer experiments. It does **not** redistribute raw speech datasets, precomputed feature caches, or model checkpoints.

Repository URL:

```text
https://github.com/amilsiddik/cross-corpus-ser-zero-target/
```

---

## 1. Study scope

This project evaluates cross-corpus SER under a strict **zero-target** setting.

In this repository, **zero-target** means that the held-out target corpus is not used for:

- training,
- validation,
- feature scaling,
- class-weight computation,
- early stopping,
- checkpoint selection,
- hyperparameter tuning,
- fine-tuning, or
- adaptation.

The held-out corpus is used only for final testing.

The analysis includes three experimental protocols:

1. **Intra-corpus evaluation** using speaker-independent train/validation/test splits.
2. **Leave-One-Dataset-Out (LODO) cross-corpus evaluation** using two corpora as sources and one corpus as the held-out target.
3. **Directional single-source to single-target transfer analysis** across all six ordered corpus pairs.

---

## 2. Datasets

The experiments use three public emotional speech corpora mapped to a shared six-class taxonomy.

| Corpus | Language | Speaking style used in this study | Classes used | Total utterances after mapping |
|---|---|---|---|---:|
| EmoDB 2.0 | German | acted utterance | angry, disgust, fear, happy, neutral, sad | 718 |
| RAVDESS | English | acted utterance | angry, disgust, fear, happy, neutral, sad | 1056 |
| RESD | Russian | improvised acted dialogue utterance | angry, disgust, fear, happy, neutral, sad | 1198 |

The unified label set is:

```text
angry, disgust, fear, happy, neutral, sad
```

Labels not consistently available across all three corpora are excluded, including:

- boredom in EmoDB 2.0,
- calm and surprised in RAVDESS,
- enthusiasm in RESD.

### Dataset availability

Raw datasets are **not included** in this repository. Users should obtain EmoDB 2.0, RAVDESS, and RESD from their original sources and follow the corresponding license and usage conditions.

The repository provides code, manifests, and result files only. It does not redistribute:

- raw audio files,
- derived audio files,
- precomputed feature caches,
- emotion2vec embeddings,
- model checkpoints.

---

## 3. Repository structure

```text
cross-corpus-ser-zero-target/
├── README.md
├── RUN_ORDER.md
├── Metadata.ipynb
├── 02_feature_extraction_handcrafted_intra.ipynb
├── 03_train_handcrafted_intra.ipynb
├── 04_train_handcrafted_mlp_intra.ipynb
├── 05_extract_emotion2vec_intra.ipynb
├── 06_train_emotion2vec_intra.ipynb
├── 07_train_concat_fusion_intra.ipynb
├── 08_train_cross_attention_fusion_intra.ipynb
├── 08b_extract_sequence_features_intra.ipynb
├── 09_train_sequence_cross_attention_intra.ipynb
├── 10_train_lodo_utterance_fusion.ipynb
├── 11_train_lodo_sequence_cross_attention.ipynb
├── 12_directional_transfer_matrix_emotion2vec.ipynb
├── Check_all.ipynb
├── plot.ipynb
├── manifests/
│   ├── environment_manifests.json
│   ├── sequence_environment_manifests.json
│   ├── generated_feature_file_manifests_sha256.csv
│   ├── generated_sequence_feature_file_manifests_sha256.csv
│   ├── lodo_split_manifests.csv
│   ├── lodo_sequence_split_manifests.csv
│   ├── lodo_leakage_prevention_checklist.csv
│   └── lodo_sequence_leakage_prevention_checklist.csv
├── results/
│   ├── compare_lodo_all_models_paper_table.csv
│   ├── directional_transfer_all_seed_results.csv
│   ├── directional_transfer_matrix_macro_f1.csv
│   ├── directional_transfer_paper_table.csv
│   ├── duration_crop_padding_stats_by_dataset.csv
│   ├── lodo_all_seed_results_with_handcrafted.csv
│   ├── lodo_paper_table_test.csv
│   ├── lodo_sequence_xattn_all_seed_results.csv
│   └── perclass_lodo_summary_mean_std_plus_base.csv
└── figures/
    ├── lodo_macro_f1_all_models_plus_base.png
    ├── lodo_macro_f1_all_models_plus_base.pdf
    └── figure3_perclass_lodo_f1_heatmap_plus_base.png
```

> Note: The current repository folder is named `manifests/`. If this folder is renamed to `manifests/`, update the paths in the README and scripts accordingly.

---

## 4. Google Drive project layout

The notebooks were developed in Google Colab and assume the following base directory:

```python
BASE_PROJECT = Path("/content/drive/MyDrive/New Jurnal Cross")
```

The original working directory contains raw datasets, processed metadata, feature outputs, model results, and figures. Raw datasets and feature caches are intentionally excluded from this GitHub repository.

Important output folders used in the experiments include:

```text
New Jurnal Cross/
├── processed_intra_csv/
├── processed_intra_features_hc_noaug/
├── processed_intra_features_e2v_plus_base/
├── processed_intra_sequence_features_plus_base/
├── results_intra_handcrafted_svm/
├── results_intra_handcrafted_mlp/
├── results_intra_emotion2vec_mlp_plus_base/
├── results_intra_concat_fusion_mlp_plus_base/
├── results_intra_sequence_cross_attention_plus_base/
├── results_lodo_utterance_fusion_plus_base/
├── results_lodo_sequence_cross_attention_plus_base/
├── results_directional_transfer_emotion2vec_plus_base/
├── revision_final_outputs/
└── figures_plus_base/
```

---

## 5. Notebook execution order

Run the main notebooks in the following order.

| Step | Notebook | Purpose | Main output |
|---:|---|---|---|
| 0 | `Metadata.ipynb` | Build metadata and speaker-independent splits | metadata and split CSV files |
| 1 | `02_feature_extraction_handcrafted_intra.ipynb` | Extract utterance-level handcrafted acoustic-prosodic features | handcrafted feature tables |
| 2 | `03_train_handcrafted_intra.ipynb` | Train Handcrafted SVM-RBF intra-corpus baseline | intra-corpus SVM results |
| 3 | `04_train_handcrafted_mlp_intra.ipynb` | Train Handcrafted MLP intra-corpus baseline | intra-corpus MLP results |
| 4 | `05_extract_emotion2vec_intra.ipynb` | Extract utterance-level emotion2vec features | 768-dimensional emotion2vec embeddings and environment manifests |
| 5 | `06_train_emotion2vec_intra.ipynb` | Train emotion2vec MLP intra-corpus model | intra-corpus emotion2vec MLP results |
| 6 | `07_train_concat_fusion_intra.ipynb` | Train utterance-level Concat Fusion MLP intra-corpus model | intra-corpus concat fusion results |
| 7 | `08b_extract_sequence_features_intra.ipynb` | Extract sequence-level emotion2vec and handcrafted features | sequence feature manifests |
| 8 | `09_train_sequence_cross_attention_intra.ipynb` | Train Sequence-level Cross-Attention intra-corpus model | intra-corpus sequence cross-attention results |
| 9 | `10_train_lodo_utterance_fusion.ipynb` | Train LODO Handcrafted SVM-RBF, Handcrafted MLP, emotion2vec MLP, and Concat Fusion MLP | LODO utterance-level results and leakage checklist |
| 10 | `11_train_lodo_sequence_cross_attention.ipynb` | Train LODO Sequence-level Cross-Attention model | LODO sequence cross-attention results and leakage checklist |
| 11 | `12_directional_transfer_matrix_emotion2vec.ipynb` | Run directional single-source to single-target transfer experiments | directional transfer matrix |
| 12 | `plot.ipynb` | Generate paper figures and final summary tables | figures and final result tables |
| 13 | `Check_all.ipynb` | Verify generated files, manifests, tables, and consistency checks | reproducibility checklist |

### Optional or legacy notebook

`08_train_cross_attention_fusion_intra.ipynb` is retained for completeness as an earlier intra-corpus cross-attention experiment. The revised manuscript focuses on the implemented **Sequence-level Cross-Attention** pipeline generated through `08b_extract_sequence_features_intra.ipynb` and `09_train_sequence_cross_attention_intra.ipynb` for intra-corpus analysis, and `11_train_lodo_sequence_cross_attention.ipynb` for LODO analysis.

---

## 6. Audio preprocessing

All audio files are standardized before feature extraction:

```text
mono
16 kHz sampling rate
fixed duration = 4 seconds
```

Longer utterances are **center-cropped**, while shorter utterances are **reflect-padded**. This produces a fixed waveform length of:

```text
4 seconds × 16,000 Hz = 64,000 samples
```

No audio augmentation is used in the final experiments.

The fixed-duration preprocessing is documented in:

```text
results/duration_crop_padding_stats_by_dataset.csv
```

Summary duration/crop/pad statistics from the final pipeline:

| Dataset | N | Mean duration (s) | Median duration (s) | Cropped (%) | Padded (%) |
|---|---:|---:|---:|---:|---:|
| EmoDB 2.0 | 718 | 2.77 | 2.57 | 9.75 | 90.25 |
| RAVDESS | 1056 | 3.72 | 3.67 | 18.75 | 81.25 |
| RESD | 1198 | 6.10 | 5.15 | 59.60 | 40.40 |

Because the fixed 4-second crop/pad procedure affects corpora differently, it should be interpreted as a documented preprocessing choice and not as a neutral corpus-independent transformation.

---

## 7. Feature representations

### 7.1 Utterance-level handcrafted features

The utterance-level handcrafted acoustic-prosodic vector has **548 dimensions** and is used by:

- Handcrafted SVM-RBF,
- Handcrafted MLP,
- Concat Fusion MLP.

The 548-dimensional vector contains:

| Feature group | Dimension |
|---|---:|
| MFCC mean + standard deviation | 80 |
| Delta MFCC mean + standard deviation | 80 |
| Delta-delta MFCC mean + standard deviation | 80 |
| Chroma mean + standard deviation | 24 |
| Log-Mel statistics | 256 |
| Spectral contrast mean + standard deviation | 14 |
| Spectral rolloff mean + standard deviation | 2 |
| Spectral bandwidth mean + standard deviation | 2 |
| Prosodic block: F0, jitter approximation, RMS, ZCR | 10 |
| **Total** | **548** |

### 7.2 Utterance-level emotion2vec features

The emotion2vec extractor produces one **768-dimensional embedding** per utterance:

```text
1 utterance → 768-dimensional emotion2vec embedding
```

This representation is used by:

- emotion2vec MLP,
- Concat Fusion MLP.

### 7.3 Sequence-level features

For Sequence-level Cross-Attention, each utterance is represented as a fixed-length temporal sequence:

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

The 548-dimensional handcrafted vector and the `T × 43` handcrafted sequence are intentionally different. The 548-dimensional vector summarizes the whole utterance using statistical functionals, whereas the `T × 43` representation preserves frame-level temporal information for attention-based fusion.

---

## 8. emotion2vec checkpoint and environment

The final experiments use the following public FunASR/ModelScope checkpoint:

```text
Model ID: iic/emotion2vec_plus_base
Local checkpoint file: model.pt
Checkpoint SHA256: 60710b5aae1dbe69bdac8920028fb05882d4314fd09031922b4b61ee9e7aadbd
FunASR version: 1.4.1
ModelScope version: 1.39.1
```

Environment details and checkpoint verification are recorded in:

```text
manifests/environment_manifests.json
manifests/sequence_environment_manifests.json
manifests/generated_feature_file_manifests_sha256.csv
manifests/generated_sequence_feature_file_manifests_sha256.csv
```

The public emotion2vec checkpoint is used as a **frozen feature extractor**. Its parameters are not updated during downstream classifier training.

The downstream experiments are zero-target with respect to EmoDB 2.0, RAVDESS, and RESD. However, the original pre-training or fine-tuning composition of the public checkpoint is outside the control of this repository. In particular, the high held-out RAVDESS performance may partly reflect alignment between RAVDESS, as the only English target corpus in this study, and the language composition of the public emotion2vec pre-training/fine-tuning data. This remains an unresolved confounding factor.

---

## 9. Experimental protocols

### 9.1 Intra-corpus evaluation

Each corpus is split into train, validation, and test sets using a speaker-independent protocol.

Target split ratio:

```text
train / validation / test ≈ 70 / 15 / 15
```

The final utterance ratio may vary because speakers are not allowed to appear in more than one split and each split is constrained to contain all six emotion classes.

### 9.2 Zero-target LODO evaluation

LODO means **Leave-One-Dataset-Out**. In each fold, two corpora are used as source corpora and the remaining corpus is held out entirely for final testing.

| Held-out target corpus | Source corpora | Corpus-pair characterization |
|---|---|---|
| EmoDB 2.0 | RAVDESS + RESD | English/Russian source corpora to German target corpus; source side includes acted utterances and improvised acted dialogue |
| RAVDESS | EmoDB 2.0 + RESD | German/Russian source corpora to English target corpus; source side includes acted utterances and improvised acted dialogue |
| RESD | EmoDB 2.0 + RAVDESS | German/English acted utterance source corpora to Russian improvised acted dialogue target corpus |

In each LODO fold:

- source train and source test partitions are merged for training,
- source validation partitions are used for early stopping and model selection,
- the entire held-out target corpus is used only for final testing.

The held-out target corpus is not used for training, validation, feature scaling, class-weight computation, early stopping, checkpoint selection, hyperparameter tuning, fine-tuning, or adaptation.

### 9.3 Directional transfer analysis

Directional transfer uses a single source corpus and a single target corpus. The evaluated directions are:

```text
EmoDB 2.0 → RAVDESS
RAVDESS → EmoDB 2.0
EmoDB 2.0 → RESD
RAVDESS → RESD
RESD → EmoDB 2.0
RESD → RAVDESS
```

This analysis characterizes asymmetric corpus-pair transfer patterns. It does not causally isolate language, corpus, recording, labeling, emotion-distribution, or speaking-style effects.

---

## 10. Models

Five models are evaluated.

| Model | Input | Purpose |
|---|---|---|
| Handcrafted SVM-RBF | 548-dimensional handcrafted vector | classical handcrafted-feature baseline |
| Handcrafted MLP | 548-dimensional handcrafted vector | neural handcrafted-feature baseline |
| emotion2vec MLP | 768-dimensional emotion2vec embedding | frozen pre-trained representation baseline |
| Concat Fusion MLP | 768-dimensional emotion2vec embedding + 548-dimensional handcrafted vector | naive utterance-level fusion |
| Sequence-level Cross-Attention | `T × 768` emotion2vec sequence + `T × 43` handcrafted sequence | sequence-level temporal fusion |

The fusion comparison should be interpreted as a comparison of **heterogeneous implemented model configurations**, not as a controlled ablation of a single architectural factor. The Concat Fusion MLP and Sequence-level Cross-Attention model differ in input structure, temporal resolution, pooling strategy, classifier configuration, and training conditions.

---

## 11. Model hyperparameters

### 11.1 Handcrafted SVM-RBF

```python
SVC(
    kernel="rbf",
    C=10.0,
    gamma="scale",
    class_weight="balanced",
    probability=True,
    random_state=seed,
)
```

### 11.2 Handcrafted MLP

```text
Input: 548
Architecture: 548 → 256 → 128 → 6
Normalization: BatchNorm1d
Activation: ReLU
Dropout: 0.30
```

### 11.3 emotion2vec MLP

```text
Input: 768
Architecture: 768 → 256 → 128 → 6
Normalization: LayerNorm
Activation: ReLU
Dropout: 0.30
```

### 11.4 Concat Fusion MLP

```text
Input: 1316
Architecture: 1316 → 512 → 256 → 128 → 6
Normalization: LayerNorm
Activation: ReLU
Dropout: 0.35
```

### 11.5 Sequence-level Cross-Attention

```text
Input emotion2vec stream: T × 768
Input handcrafted stream: T × 43
T: 200 frames
d_model: 128
attention heads: 4
attention dropout: 0.30
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

Because each component is 128-dimensional:

```text
8 × 128 = 1024-dimensional fused vector
```

Classifier:

```text
1024 → 512 → 256 → 6
LayerNorm + ReLU + Dropout(0.30)
```

---

## 12. Training hyperparameters

All neural models use:

```text
optimizer: AdamW
weight decay: 1e-4
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

For each seed, the checkpoint with the highest validation Macro-F1 is selected. Test scores from the three seeds are then summarized as mean and standard deviation.

The reported standard deviations reflect variability across random seeds under fixed dataset splits. They do not represent sampling uncertainty over corpora, speakers, or utterances.

---

## 13. Evaluation metrics

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
mean ± standard deviation over three random seeds
```

---

## 14. Main result files

The main CSV result files are stored in `results/`.

| File | Description |
|---|---|
| `results/lodo_all_seed_results_with_handcrafted.csv` | seed-level LODO results for utterance-level models including handcrafted baselines |
| `results/lodo_paper_table_test.csv` | LODO table for utterance-level models |
| `results/lodo_sequence_xattn_all_seed_results.csv` | seed-level LODO results for Sequence-level Cross-Attention |
| `results/compare_lodo_all_models_paper_table.csv` | final LODO comparison across all five evaluated models |
| `results/directional_transfer_all_seed_results.csv` | seed-level directional transfer results |
| `results/directional_transfer_paper_table.csv` | directional transfer table used in the manuscript |
| `results/directional_transfer_matrix_macro_f1.csv` | directional Macro-F1 matrix |
| `results/perclass_lodo_summary_mean_std_plus_base.csv` | per-class LODO F1 mean ± standard deviation |
| `results/duration_crop_padding_stats_by_dataset.csv` | duration, crop, and padding statistics by dataset |

---

## 15. Main experimental findings

The following findings should be interpreted descriptively under the present three-corpus zero-target protocol.

1. **emotion2vec MLP achieved the highest average LODO Macro-F1 among the evaluated models.**
   - Average LODO Macro-F1: **79.12%**

2. **Handcrafted-only baselines showed substantially lower zero-target cross-corpus performance.**
   - Handcrafted SVM-RBF average LODO Macro-F1: **18.83%**
   - Handcrafted MLP average LODO Macro-F1: **19.61%**

3. **Naive utterance-level concatenation did not improve over emotion2vec MLP.**
   - emotion2vec MLP average LODO Macro-F1: **79.12%**
   - Concat Fusion MLP average LODO Macro-F1: **77.63%**

4. **Sequence-level Cross-Attention slightly outperformed Concat Fusion MLP but did not surpass emotion2vec MLP.**
   - Sequence-level Cross-Attention average LODO Macro-F1: **78.39%**
   - Concat Fusion MLP average LODO Macro-F1: **77.63%**
   - This comparison should be interpreted as a comparison of heterogeneous implemented model configurations, not as a controlled ablation of the attention mechanism.

5. **RESD produced the lowest absolute LODO Macro-F1 among the three held-out corpora.**
   - This result should be interpreted as a corpus-pair effect under jointly varying language, corpus construction, recording conditions, label procedures, emotion distribution, and speaking style, rather than as causal evidence that dialogue style alone is more difficult than language shift.

---

## 16. Directional transfer results

Directional transfer was evaluated using emotion2vec MLP.

| Source | Target | Macro-F1 (%) |
|---|---|---:|
| EmoDB 2.0 | RAVDESS | 94.24 |
| RAVDESS | EmoDB 2.0 | 85.18 |
| EmoDB 2.0 | RESD | 58.89 |
| RAVDESS | RESD | 61.84 |
| RESD | EmoDB 2.0 | 83.18 |
| RESD | RAVDESS | 93.91 |

These results show asymmetric corpus-pair transfer patterns. They do not establish causal separation of language and speaking-style effects.

---

## 17. Figures

The main figures generated for the manuscript are stored in `figures/`.

| File | Description |
|---|---|
| `figures/lodo_macro_f1_all_models_plus_base.png` | LODO Macro-F1 comparison across held-out corpora and models |
| `figures/lodo_macro_f1_all_models_plus_base.pdf` | PDF version of the LODO Macro-F1 figure |
| `figures/figure3_perclass_lodo_f1_heatmap_plus_base.png` | Per-class LODO F1 heatmap across held-out corpora and emotion2vec-based models |

---

## 18. Leakage-prevention and reproducibility artifacts

The repository includes split manifests and leakage-prevention checklists in `manifests/`.

| File | Purpose |
|---|---|
| `manifests/lodo_split_manifests.csv` | train/validation/target-test manifests for LODO utterance-level experiments |
| `manifests/lodo_sequence_split_manifests.csv` | train/validation/target-test manifests for LODO sequence-level experiments |
| `manifests/lodo_leakage_prevention_checklist.csv` | leakage-prevention checklist for LODO utterance-level experiments |
| `manifests/lodo_sequence_leakage_prevention_checklist.csv` | leakage-prevention checklist for LODO sequence-level experiments |
| `manifests/generated_feature_file_manifests_sha256.csv` | SHA256 manifests for generated utterance-level emotion2vec feature files |
| `manifests/generated_sequence_feature_file_manifests_sha256.csv` | SHA256 manifests for generated sequence-level feature files |
| `manifests/environment_manifests.json` | environment and checkpoint manifests for utterance-level emotion2vec extraction |
| `manifests/sequence_environment_manifests.json` | environment and checkpoint manifests for sequence-level feature extraction |

The leakage-prevention checks are intended to document that target-corpus information is not used during source training or model selection. Feature caches are treated as per-utterance representations; no corpus-level normalization statistics, feature-selection criteria, class weights, validation scores, or target labels from the held-out corpus should be used during source training or model selection.

---

## 19. Installation

The notebooks were run in Google Colab. Exact versions are recorded in `manifests/environment_manifests.json` and `manifests/sequence_environment_manifests.json`.

A minimal installation example is:

```bash
pip install numpy pandas scikit-learn librosa soundfile matplotlib seaborn tqdm openpyxl
pip install torch torchaudio
pip install funasr==1.4.1 modelscope==1.39.1
```

If running on Google Colab, mount Google Drive first:

```python
from google.colab import drive
drive.mount('/content/drive')
```

Then set the project directory in the notebooks:

```python
from pathlib import Path
BASE_PROJECT = Path("/content/drive/MyDrive/New Jurnal Cross")
```

---

## 20. Recommended `.gitignore`

The following file types and directories should not be committed to the public repository:

```gitignore
# Raw datasets and audio files
datasets/
data/
raw_audio/
*.wav
*.mp3
*.flac
*.m4a

# Feature caches and derived embeddings
processed_features/
feature_cache/
emotion2vec_cache/
*.npy
*.npz
*.pkl
*.joblib

# Model checkpoints
checkpoints/
models/
*.pt
*.pth
*.ckpt
*.bin

# Secrets
.env
*.key
*.pem
secrets.json

# Python and notebook temporary files
__pycache__/
*.pyc
.ipynb_checkpoints/

# System files
.DS_Store
Thumbs.db
```

---

## 21. Data and code availability statement

Suggested manuscript statement:

```text
The experimental code, configuration files, split manifests, fold-level result files, and figure-generation scripts are available at https://github.com/amilsiddik/cross-corpus-ser-zero-target/. The raw speech datasets and precomputed feature caches are not redistributed and should be obtained or generated by users from the original datasets under their respective license and usage conditions. The repository provides instructions for reproducing the zero-target LODO and directional transfer experiments using the same preprocessing, feature extraction, and model-selection procedures.
```

---

## 22. Limitations of this repository

This repository supports verification of the experimental pipeline and reported results, but it has the following limitations:

- Raw datasets are not redistributed.
- Precomputed feature caches are not redistributed.
- Model checkpoints are not redistributed.
- Full reproduction requires users to obtain the datasets from their original sources.
- Results depend on the public `iic/emotion2vec_plus_base` checkpoint and the software environment recorded in the manifests files.
- The reported standard deviations reflect random-seed variability under fixed splits, not uncertainty over alternative corpora, speakers, or utterances.

---

## 23. License and dataset usage

This repository is provided for research reproducibility. Users must comply with the licenses and usage conditions of EmoDB 2.0, RAVDESS, RESD, emotion2vec, FunASR, ModelScope, librosa, scikit-learn, and PyTorch.

Raw datasets should be downloaded only from their official or original distribution sources.

---

## 24. Contact

For questions about the code or experiments, please contact the corresponding author of the manuscript.

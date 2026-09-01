# Experiment B Run Order: plus_base Zero-Target Cross-Corpus Experiments

This document describes the execution order used for the `plus_base` zero-target cross-corpus speech emotion recognition experiments.

The main working folder in Google Colab was:

```python
from pathlib import Path
BASE_PROJECT = Path("/content/drive/MyDrive/New Jurnal Cross")
````

The emotion2vec checkpoint used in the final experiments was:

```python
CONFIG["model_name"] = "iic/emotion2vec_plus_base"
```

The output folders use the suffix `plus_base` so that the new experiment does not overwrite previous experiment folders.

## Execution Order

The notebooks should be executed in the following order:

1. `Metadata.ipynb`
2. `02_feature_extraction_handcrafted_intra.ipynb`
3. `03_train_handcrafted_intra.ipynb`
4. `04_train_handcrafted_mlp_intra.ipynb`
5. `05_extract_emotion2vec_intra.ipynb`
6. `06_train_emotion2vec_intra.ipynb`
7. `07_train_concat_fusion_intra.ipynb`
8. `08b_extract_sequence_features_intra.ipynb`
9. `09_train_sequence_cross_attention_intra.ipynb`
10. `10_train_lodo_utterance_fusion.ipynb`
11. `11_train_lodo_sequence_cross_attention.ipynb`
12. `12_directional_transfer_matrix_emotion2vec.ipynb`
13. `plot.ipynb`
14. `Check_all.ipynb`

Note: `08_train_cross_attention_fusion_intra.ipynb` was also fixed, but this model is not a primary reported model if the manuscript reports only the Sequence-level Cross-Attention model.

## Working Input and Output Folders in Google Drive

The following folders were used during execution in Google Colab.

### Input/cache folders

* `processed_intra_features_hc_noaug`
* `processed_intra_features_e2v_plus_base`
* `processed_intra_sequence_features_plus_base`

### Main result folders

* `results_intra_handcrafted_svm_plus_base`
* `results_intra_handcrafted_mlp_plus_base`
* `results_intra_emotion2vec_mlp_plus_base`
* `results_intra_concat_fusion_mlp_plus_base`
* `results_intra_sequence_cross_attention_plus_base`
* `results_lodo_utterance_fusion_plus_base`
* `results_lodo_sequence_cross_attention_plus_base`
* `results_directional_transfer_emotion2vec_plus_base`

## GitHub Repository Structure

The public GitHub repository does not redistribute raw speech datasets, precomputed feature caches, or trained model checkpoints. Instead, it provides notebooks, manifests, summary results, and figure-generation files needed to verify and reproduce the reported experiments.

Final copies of the most important manifest and result files are provided in the `manifests/`, `results/`, and `figures/` folders.

```text
cross-corpus-ser-zero-target/
├── README.md
├── RUN_ORDER.md
├── requirements.txt
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
├── results/
└── figures/
```

## Important Files for Revision and Reviewer Response

### Feature and environment manifests

The final GitHub copies are stored in `manifests/`:

* `manifests/environment_manifest.json`
* `manifests/sequence_environment_manifest.json`
* `manifests/generated_feature_file_manifest_sha256.csv`
* `manifests/generated_sequence_feature_file_manifest_sha256.csv`

These files correspond to the working outputs originally generated in:

* `processed_intra_features_e2v_plus_base/environment_manifest.json`
* `processed_intra_features_e2v_plus_base/generated_feature_file_manifest_sha256.csv`
* `processed_intra_sequence_features_plus_base/sequence_environment_manifest.json`
* `processed_intra_sequence_features_plus_base/generated_sequence_feature_file_manifest_sha256.csv`

### LODO split and leakage documentation

The final GitHub copies are stored in `manifests/`:

* `manifests/lodo_split_manifest.csv`
* `manifests/lodo_split_summary.csv`
* `manifests/lodo_partition_overlap_check.csv`
* `manifests/lodo_leakage_prevention_checklist.csv`
* `manifests/lodo_sequence_split_manifest.csv`
* `manifests/lodo_sequence_split_summary.csv`
* `manifests/lodo_sequence_leakage_prevention_checklist.csv`

These files document the LODO fold construction, source/target separation, and leakage-prevention checks.

### Main paper result tables

The final GitHub copies are stored in `results/`:

* `results/lodo_paper_table_test.csv`
* `results/compare_lodo_all_models_paper_table.csv`
* `results/perclass_lodo_summary_mean_std_plus_base.csv`
* `results/duration_crop_padding_stats_by_dataset.csv`

### Seed-level result files

The final GitHub copies are stored in `results/`:

* `results/lodo_all_seed_results_with_handcrafted.csv`
* `results/lodo_sequence_xattn_all_seed_results.csv`
* `results/directional_transfer_all_seed_results.csv`

These files provide seed-wise validation and target-test results used to compute the reported mean ± standard deviation values.

### Directional transfer results

The final GitHub copies are stored in `results/`:

* `results/directional_transfer_all_seed_results.csv`
* `results/directional_transfer_summary_mean_std.csv`
* `results/directional_transfer_matrix_macro_f1.csv`
* `results/directional_transfer_paper_table.csv`

The file `directional_transfer_summary_mean_std.csv` contains the mean ± standard deviation values for the six single-source to single-target directional transfer pairs.

### Figures

The final GitHub copies are stored in `figures/`:

* `figures/lodo_macro_f1_all_models_plus_base.png`
* `figures/lodo_macro_f1_all_models_plus_base.pdf`
* `figures/figure3_perclass_lodo_f1_heatmap_plus_base.png`

## Notes on Raw Data, Feature Caches, and Checkpoints

The repository does not include:

* raw speech audio files,
* precomputed feature caches,
* trained model checkpoints,
* API keys,
* credentials,
* temporary runtime files.

Users should obtain the original datasets from their respective sources and regenerate the features and results by following the notebook execution order above.

## Notes on Terminology

In the revised manuscript, EmoDB 2.0 and RAVDESS are described as acted utterance corpora, while RESD is described as an improvised acted dialogue corpus or actor-voiced emotional dialogue corpus. If older internal metadata fields use the term `monologue`, they should be interpreted only as legacy internal labels for isolated acted utterance corpora, not as manuscript terminology.


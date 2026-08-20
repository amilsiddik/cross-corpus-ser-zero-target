# Experiment B Run Order: plus_base Zero-Target Cross-Corpus Experiments

Main Colab folder:

```python
BASE_PROJECT = Path("/content/drive/MyDrive/New Jurnal Cross")
```

Model used:

```python
CONFIG["model_name"] = "iic/emotion2vec_plus_base"
```

The output folders use the suffix `plus_base` so that the new experiment does not overwrite previous experiment folders.

## Execution Order

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

Note: `08_train_cross_attention_fusion_intra.ipynb` was also fixed, but this model is not a priority if the manuscript only reports the Sequence-level Cross-Attention model.

## Main Input and Output Folders

Input/cache folders:

- `processed_intra_features_hc_noaug`
- `processed_intra_features_e2v_plus_base`
- `processed_intra_sequence_features_plus_base`

Main result folders:

- `results_intra_handcrafted_svm_plus_base`
- `results_intra_handcrafted_mlp_plus_base`
- `results_intra_emotion2vec_mlp_plus_base`
- `results_intra_concat_fusion_mlp_plus_base`
- `results_intra_sequence_cross_attention_plus_base`
- `results_lodo_utterance_fusion_plus_base`
- `results_lodo_sequence_cross_attention_plus_base`

## Important Files for Revision and Reviewer Response

Feature and environment manifests:

- `processed_intra_features_e2v_plus_base/environment_manifest.json`
- `processed_intra_features_e2v_plus_base/generated_feature_file_manifest_sha256.csv`
- `processed_intra_sequence_features_plus_base/sequence_environment_manifest.json`
- `processed_intra_sequence_features_plus_base/generated_sequence_feature_file_manifest_sha256.csv`
- `results_lodo_utterance_fusion_plus_base/training_environment_manifest.json`
- `results_lodo_sequence_cross_attention_plus_base/sequence_training_environment_manifest.json`

LODO split and leakage documentation:

- `results_lodo_utterance_fusion_plus_base/lodo_split_manifest.csv`
- `results_lodo_utterance_fusion_plus_base/lodo_split_summary.csv`
- `results_lodo_utterance_fusion_plus_base/lodo_leakage_prevention_checklist.csv`
- `results_lodo_utterance_fusion_plus_base/lodo_partition_overlap_check.csv`
- `results_lodo_sequence_cross_attention_plus_base/lodo_sequence_split_manifest.csv`
- `results_lodo_sequence_cross_attention_plus_base/lodo_sequence_split_summary.csv`
- `results_lodo_sequence_cross_attention_plus_base/lodo_sequence_leakage_prevention_checklist.csv`

Main paper tables:

- `results_lodo_utterance_fusion_plus_base/lodo_paper_table_test.csv`
- `results_lodo_sequence_cross_attention_plus_base/compare_lodo_all_models_paper_table.csv`

Seed-level result files:

- `results_lodo_utterance_fusion_plus_base/lodo_all_seed_results_with_handcrafted.csv`
- `results_lodo_sequence_cross_attention_plus_base/lodo_sequence_xattn_all_seed_results.csv`

Directional transfer results:

- `results_directional_transfer_emotion2vec_plus_base/directional_transfer_matrix_macro_f1.csv`
- `results_directional_transfer_emotion2vec_plus_base/directional_transfer_paper_table.csv`
- `results_directional_transfer_emotion2vec_plus_base/directional_transfer_all_seed_results.csv`

Figures:

- `figures/lodo_macro_f1_all_models_plus_base.png`
- `figures/lodo_macro_f1_all_models_plus_base.pdf`
- `figures/figure3_perclass_lodo_f1_heatmap_plus_base.png`
- `figures/figure3_perclass_lodo_f1_heatmap_plus_base.pdf`


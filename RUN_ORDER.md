# ExperimentB run order: plus_base zero-target cross-corpus experiments

Folder utama Colab:

```python
BASE_PROJECT = Path("/content/drive/MyDrive/New Jurnal Cross")
```

Model emotion2vec yang digunakan untuk revisi:

```python
CONFIG["model_name"] = "iic/emotion2vec_plus_base"
```

Output baru diberi suffix `plus_base` agar tidak menimpa folder lama.

## Urutan eksekusi

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

`08_train_cross_attention_fusion_intra.ipynb` diperbaiki juga, tetapi model tersebut bukan prioritas utama jika manuscript hanya memakai Sequence-level Cross-Attention.

## Folder output utama

- `processed_intra_features_e2v_plus_base`
- `processed_intra_sequence_features_plus_base`
- `results_intra_emotion2vec_mlp_plus_base`
- `results_intra_concat_fusion_mlp_plus_base`
- `results_intra_sequence_cross_attention_plus_base`
- `results_lodo_utterance_fusion_plus_base`
- `results_lodo_sequence_cross_attention_plus_base`

## File penting untuk revisi/reviewer

- `processed_intra_features_e2v_plus_base/environment_manifest.json`
- `processed_intra_features_e2v_plus_base/generated_feature_file_manifest_sha256.csv`
- `processed_intra_sequence_features_plus_base/sequence_environment_manifest.json`
- `processed_intra_sequence_features_plus_base/generated_sequence_feature_file_manifest_sha256.csv`
- `results_lodo_utterance_fusion_plus_base/lodo_paper_table_test.csv`
- `results_lodo_sequence_cross_attention_plus_base/compare_lodo_all_models_paper_table.csv`

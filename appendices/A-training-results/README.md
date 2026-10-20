# Appendix A: Training and Validation Results

Final training runs (April 2026) on the 1,632,973-example corpus (70% benign / 30% injection), split 80/20: 1,306,378 training and 326,595 held-out validation examples (227,572 benign).

Eight classical classifiers were trained on three feature representations (TF-IDF, MiniLM, mpnet) under three balancing conditions (none, class weight, SMOTE; SMOTE on dense embeddings only), and two transformers (BERT Base Uncased, SecureBERT 2.0) were fine-tuned end to end. That is 66 executed runs. Naïve Bayes does not support class weights, so its two "class weight" runs are identical to "none", leaving 64 distinct configurations.

| File | Contents |
|---|---|
| `A1_validation_mcc_all_runs.png` | Validation MCC for every classical run (model × feature × balancing) |
| `A2_learning_curves_tfidf.png` | Learning curves, all 8 classical models, TF-IDF |
| `A3_learning_curves_minilm.png` | Learning curves, all 8 classical models, MiniLM |
| `A4_learning_curves_mpnet.png` | Learning curves, all 8 classical models, mpnet |
| `A5_learning_curves_transformers.png` | Learning curves, BERT Base Uncased and SecureBERT 2.0 |
| `A6_training_history_bert_base.png`, `A7_training_history_securebert2.png` | Per-epoch training history saved during fine-tuning |
| `training_results_all_runs.csv` | Every run: MCC, F1, precision, recall, false alarm rate, accuracy, confusion matrix, training time |

These results correspond to Tables 9 and 10 and Figures 15–19 of the praxis.

# Appendix B: Dataset Composition

| File | Contents |
|---|---|
| `B1_corpus_composition.csv` | Final corpus (1,632,973 examples) and the stratified 80/20 train/validation split (Table B1) |
| `B2_attack_surface_alignment.csv` | Injection attack surfaces and aligned benign domains (Table B2) |

Validation counts come from the validation confusion matrices; training counts are the full corpus minus validation. Injection counts per attack surface are approximate and measured before deduplication; the final corpus contains 495,112 unique injection examples.

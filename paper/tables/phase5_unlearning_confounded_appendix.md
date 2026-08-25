# Appendix: the ORIGINAL confounded class-scenario forgetting metric — DOCUMENTED NEGATIVE EXAMPLE, NOT EVIDENCE

Preserved per the project brief ('Retain the original confounded result only as a documented negative example
in an appendix, not as evidence.'). This compares removed class-level data against retained TEST DATA OF
OTHER CLASSES -- confounded because any class-level loss asymmetry that exists even in an UNTRAINED model
biases this number regardless of real memorization (e.g. checkpoint_rollback, never trained on the
post-removal data at all, still showed ~0.99 here in an earlier run). Do not cite this table as evidence of
forgetting quality; see phase5_unlearning_synthetic.md for the corrected within-class primary metric.

| method | n_seeds | confounded symmetric AUC |
|---|---|---|
| adapter_deletion_and_retrain | 3 | 1.000 ± 0.000 |
| checkpoint_rollback | 3 | 0.988 ± 0.021 |
| full_retrain | 3 | 1.000 ± 0.000 |
| gradient_ascent_unlearning | 3 | 1.000 ± 0.000 |
| key_revocation_only | 3 | 0.821 ± 0.215 |

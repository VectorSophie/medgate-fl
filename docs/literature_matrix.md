# Literature matrix

Authoritative machine-readable data: `docs/literature_matrix.csv` (17
columns: id, title, authors, year, venue, doi_or_arxiv, peer_review_status,
problem_addressed, threat_model, dataset, baselines, metrics,
code_availability, relation_to_this_work, verification_status,
verification_source, notes). This file is a grouped, human-readable index
into it — **the CSV is authoritative**; if the two ever disagree, trust the
CSV and fix this file.

Current coverage (2026-08-25): **59 rows, 16 `VERIFIED`, 43
`UNVERIFIED`**. Only `VERIFIED` rows may be cited in `paper/main.tex`
(rule from the project brief: unverifiable references are marked
`UNVERIFIED` and excluded from the paper until verified). Verification in
this session means: the title/authors/venue/identifier were checked against
a primary source (arXiv abstract page, ACM DL/USENIX/DBLP record, Crossref
DOI lookup, or the FLamby/ISIC official repositories/pages) — not that the
paper's *content* was read in full. Verifying the remaining 43 rows is
listed as an explicit open task in `docs/execution_plan.md`, not a hidden
gap.

## Federated learning (core optimization)
`mcmahan2017-fedavg` (V), `li2020-fedprox` (V), `karimireddy2020-scaffold`
(V), `reddi2021-fedadam` (V), `kairouz2021-advances` (V),
`li2020-fl-challenges` (U).
**Gap this project addresses:** these papers optimize *for* heterogeneous,
decentralized data; none of them separate a public-safe capability from a
restricted one within a single federated model. MedGate-FL treats that
separation as the object of study, using FedAvg/FedProx/SCAFFOLD/FedAdam
only as the aggregation substrate and as baselines to beat or fail to beat.

## Medical federated learning and benchmarks
`duterrail2022-flamby` (V), `teo2024-fl-healthcare-review` (U),
`sheller2020-fl-medicine` (U), `rieke2020-digital-health-fl` (U),
`pati2021-fets` (U), `stripelis2024-neuroimaging-fl` (U),
`yang2023-medmnistv2` (U).
**Gap:** FLamby supplies realistic natural splits and honest baselines but
does not study access tiering or capability isolation within a trained
model; the clinical-FL surveys argue FL helps locality/collaboration, not
that it protects a model's outputs from an unauthorized holder of that
model. This project's non-claims (§3 of `docs/research_scope.md`) exist
specifically to avoid inheriting that conflation.

## Parameter-efficient adaptation and model protection
`hu2022-lora` (V), `ahn2026-lorenc` (V), `mia2025-fedshield-llm` (U),
`rho2024-encryption-friendly-llm` (U), `narkedimilli2024-fl-dabe-bc` (U).
**Gap:** LoRA gives a cheap adapter mechanism; LoREnc gives a training-free
obfuscation transform for adapters at rest. Neither is evaluated, in its
own paper, against the specific attacker set (A1-A4) or the federated
cross-silo setting this project uses. MedGate-FL's job is to test whether
LoRA-as-capability-boundary plus LoREnc-style obfuscation actually survives
those attacks here — not to assume it does because the pieces exist.

## Secure aggregation and differential privacy
`bonawitz2017-secagg` (V), `abadi2016-dpsgd` (V),
`dwork2014-algorithmic-foundations` (U), `geyer2017-dp-fl-client-level` (U),
`mcmahan2018-dp-rnn` (U), `mohamad-sok-secagg` (U).
**Gap:** secure aggregation hides individual updates from the server;
DP-SGD bounds what any single output can reveal. Neither, by itself, is a
capability-isolation mechanism, and the project brief explicitly warns
against claiming secure aggregation prevents membership/property inference
— Phase 4 measures that boundary rather than assuming it.

## Inference and extraction attacks
`zhu2019-dlg` (V), `zhao2020-idlg` (U), `geiping2020-inverting-gradients`
(U), `shokri2017-mia` (V), `nasr2019-comprehensive-privacy` (U),
`melis2019-unintended-leakage` (U), `fredrikson2015-model-inversion` (U),
`hitaj2017-gan-leakage` (U), `carlini2021-extracting-training-data` (U),
`tramer2016-modelextraction` (V).
**Gap:** these attacks target centralized or generically federated models;
this project adapts them specifically to ask whether the *public* path of a
capability-isolated model leaks the *restricted* task, which is a different
question from "does FL leak training data" in general.

## Access control and cryptography
`bethencourt2007-cpabe` (U), `goyal2006-abe` (U),
`goldwasser1984-probabilistic-encryption` (U), `gentry2009-fhe` (U),
`cheon2017-ckks` (U).
**Gap:** these are foundational constructions, not medical-FL-specific;
they inform why this project keeps cryptographic claims conservative
(AES-GCM for adapters at rest, no custom crypto, RSA never used to encrypt
tensors directly) rather than reaching for FHE/ABE machinery this
project's hardware and timeline cannot properly evaluate.

## Blockchain-assisted FL
`blockchain-fl-healthcare-review-1` (U), `blockchain-fl-healthcare-review-2`
(U, placeholder — see notes column in the CSV), `blockchain-fl-slr` (U).
**Gap:** the project brief itself requires including *negative* and
critical literature on blockchain necessity/scalability/GDPR conflicts;
none of these three rows is verified yet, so no blockchain-necessity claim
of any kind appears in `paper/main.tex` until at least one is.

## Federated and medical unlearning
`liu-federated-unlearning-ambiguous` (U, deliberately unresolved — see
notes), `wu2022-fed-unlearning-kd` (U), `halimi-fed-unlearning-erase-client`
(U), `zhang2023-fedrecovery` (U), `deng2024-right-to-be-forgotten-medical`
(U), `chen2026-lethe-unlearning` (V), `bourtoule2021-machine-unlearning`
(U), `ginart2019-making-ai-forget` (U).
**Gap:** `chen2026-lethe-unlearning` (Lethe) is the one verified, directly
on-point prior benchmark — its finding that *forgetting-request difficulty*
rather than *method* dominates performance is a concrete prior result
Phase 5 should try to replicate or contradict, not a fact to assume.

## Datasets
`tschandl2018-ham10000` (V), `duterrail2022-flamby` (V, cross-listed above),
`ixi-official-flamby-doc` (V), `isic-archive-2021-descriptor` (U),
`milk10k-dataset-record` (U), `pad-ufes-20` (U), `torchio-paper` (U),
`fets-brats-official` (U), `chbmit-physionet` (U), `siena-eeg-physionet`
(U).
**Gap:** the primary dataset chain (FLamby, HAM10000, IXI) is verified;
every *external/fusion* dataset (ISIC 2020, MILK10k, PAD-UFES-20) and every
optional-extension dataset (FeTS/BraTS, CHB-MIT, Siena) remains unverified
and therefore unused until checked — this directly gates Phases 6-8 in
`docs/execution_plan.md`.

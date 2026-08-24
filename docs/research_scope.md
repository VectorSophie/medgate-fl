# MedGate-FL — Research Scope

Status: living document. Updated as phases complete. Last updated 2026-08-25
(Phase 0 in progress; see `docs/execution_plan.md` for phase status).

Working title: **MedGate-FL: Capability-Isolated Federated Adapters for
Tiered Medical AI Access**. Alternative title (adopt only if the final
results favor it): *Separating Public and Restricted Diagnostic Capabilities
in Federated Medical Models*. Title choice is deferred until Phase 2/3
results exist — see `docs/execution_plan.md`.

## 1. Research questions

Central question: can a federated medical model preserve useful **public,
coarse-grained** capability while isolating **fine-grained diagnostic**
capability inside a restricted adapter, such that:

- RQ1 (utility): authorized users recover nearly all fine-grained utility
  relative to a plain federated LoRA baseline;
- RQ2 (unauthorized resistance): unauthorized users cannot cheaply recover
  fine-grained capability via probing, fine-tuning, adapter reconstruction,
  collusion, or distillation, under a fixed, reported attack budget;
- RQ3 (update protection): individual hospital (client) updates receive
  meaningful, measured protection against reconstruction/inference, and this
  is evaluated against the exact observation each attacker has, not assumed;
- RQ4 (revocation vs. unlearning): access revocation (key/credential level)
  and federated unlearning (influence-removal level) are distinct
  operations with different guarantees, and the gap between them is
  measured, not asserted;
- RQ5 (authorization backend): does a minimal permissioned-ledger
  authorization prototype outperform a conventional centralized IAM
  baseline on the metrics in §7, or does it honestly fail to?

The contribution under test is **capability isolation in federated
adapters**. Blockchain, AES-GCM, ABE, secure aggregation, DP-SGD, TEEs, and
FHE are supporting/comparison mechanisms, evaluated as baselines or
components — none of them is presented as the paper's novelty.

## 2. Threat model (summary — see `docs/execution_plan.md` Phase 3 for attack
details and `paper/main.tex` §5 for the full write-up)

| ID | Adversary | Access | Goal |
|----|-----------|--------|------|
| A1 | Honest-but-curious aggregation server | Sees permitted protocol messages (client updates, or secure-agg sums depending on configuration) | Infer client data / client properties from what the protocol legitimately exposes |
| A2 | Unauthorized model recipient | Public backbone + public coarse head + encrypted/transformed restricted adapter + public auxiliary medical data + normal compute | Recover fine-grained capability without the decryption key |
| A3 | Black-box unauthorized user | Query access to an authorized inference endpoint, fixed query budget | Distill / extract fine-grained capability |
| A4 | Malicious/colluding clients | Normal client role in one or more federation rounds; may collude, poison, submit malformed updates, replay tokens | Degrade/backdoor the global model, or infer properties of other clients |

Each attack implementation in Phase 3 must declare, in its config and its
result JSON: attacker knowledge, access, query budget, compute budget, and
the exact success condition. Composite scores (ARR, UCG, RFC, CRE — §7.4)
are reported alongside raw numbers, never in place of them.

## 3. Claims and non-claims

### Claims this project is allowed to make, and only once measured
- "Capability isolation" — a measured reduction in fine-grained utility
  recoverable from the public path/representation, under stated attacks and
  budgets.
- "Restricted capability" — capability gated behind the encrypted adapter +
  authorization mechanism, as implemented.
- "Empirical attack resistance" — resistance is always relative to the
  specific, reproducible attacks in Phase 3, never general or asymptotic.
- "Simulated cross-silo evaluation" — all federated results come from
  multiple processes/partitions on shared hardware, not real institutions.

### Explicit non-claims (do not appear in `paper/main.tex` in any stronger
form than what is listed here)
- Federated learning does **not** inherently guarantee privacy.
- Public datasets (FLamby Fed-ISIC2019 etc.) do **not** prove compliance
  with real medical privacy law (HIPAA, GDPR, or otherwise).
- Encrypted adapters do **not** prevent an attacker from independently
  training another model on public data.
- LoREnc-style spectral/orthogonal transformations are **model
  protection / capability obfuscation**, not standard cryptographic
  security, unless a cited formal proof says otherwise (none is currently
  known for this construction — see `docs/literature_matrix.md`).
- Blockchain provides **no confidentiality**; it is evaluated only for
  authorization/audit properties (latency, tamper-evidence, throughput).
- Deleting a key does **not** remove a patient's learned influence from
  model weights — that is what the unlearning experiments (Phase 5) test
  separately.
- Simulated clients on one workstation are **not** equivalent to a real
  multi-hospital deployment (network, governance, data drift, staffing are
  all absent).
- The model is **not** suitable for clinical diagnosis at any point in this
  project.
- A restricted model that confidently emits *wrong* fine-grained
  predictions is **not** a successful defense — abstention/low-confidence
  behavior and wrong-but-confident behavior are reported separately.

## 4. Hardware actually available (measured 2026-08-25)

See `docs/hardware_report.md` for the machine-readable capture. Summary:

- CPU only — no NVIDIA GPU present (`nvidia-smi` not found on this
  workstation). All Phase 0/1 work is CPU-only PyTorch.
- 8 logical CPUs, 7.4 GiB total RAM, **~2.5 GiB available** at measurement
  time (the box is otherwise in use), 2 GiB swap.
- 58 GiB free disk under the working directory.
- Outbound network access confirmed to github.com, arxiv.org, PyPI.

Consequences for scope, stated up front rather than discovered mid-run:

- Full Fed-ISIC2019 (23,247 images, 224px, 6 centers) training with a
  ResNet-sized backbone across the full baseline+ablation+attack matrix in
  the spec is **not feasible on this hardware** in a reasonable session
  wall-clock budget. This is a resource constraint, not a scope
  disagreement — it is recorded here and in every results table where it
  binds, rather than silently downscaling and presenting numbers as if they
  were the full-scale result.
- Phase 0 and Phase 1 "smoke" tiers therefore run on (a) fully synthetic
  fixtures and (b) — once the license-gated real download is performed by a
  human — a small deterministic subset of real Fed-ISIC2019, with a tiny
  CNN backbone (not a pretrained ResNet/EfficientNet), documented batch
  sizes, and wall-clock budgets per run.
- SCAFFOLD, DP-SGD sweeps, gradient-inversion attacks, and any
  FeTS/BraTS/ADNI/EEG work are gated behind Phase 1 completing and an
  explicit resource re-check (Phase 8 rule in `docs/execution_plan.md`).
- FHE and full blockchain-network experiments are out of scope for this
  hardware; only the minimal permissioned-ledger simulation described in
  the spec is attempted, and only after Phases 0–5 are done.

## 5. Primary dataset — FLamby Fed-ISIC2019 (verified 2026-08-25)

Verified directly from the official FLamby repository
(`github.com/owkin/FLamby`, files fetched: top-level `README.md`,
`flamby/datasets/fed_isic2019/README.md`, `.../dataset.py`,
`.../common.py`) and the ISIC Challenge data page
(`challenge.isic-archive.com/data/`). See `docs/literature_matrix.csv` rows
`flamby-2022`, `ham10000-2018` for full citation metadata.

| Property | Verified value | Source |
|---|---|---|
| Images | 23,247 skin-lesion images with identifiable source center | FLamby fed_isic2019 README |
| Centers | 6: BCN, HAM_vidir_molemax, HAM_vidir_modern, HAM_rosendahl, MSK, HAM_vienna_dias | FLamby fed_isic2019 README |
| Task | Multiclass classification, 8 fine-grained classes: MEL, NV, BCC, AK, BKL, DF, VASC, SCC | ISIC 2019 challenge data page + corroborating literature (dataset.py itself does not hardcode the label list — see note below) |
| Split | Predetermined, center-stratified train/test split (`fold2` column, `train_X`/`test_X`) | FLamby `dataset.py` |
| License | **CC BY-NC 4.0** for ISIC2019 imagery; HAM10000-sourced portion carries its own Harvard Dataverse terms and must be cited via Tschandl et al. 2018 (DOI 10.1038/sdata.2018.161) | ISIC Challenge data page, FLamby fed_isic2019 README |
| Download | `python flamby/datasets/fed_isic2019/dataset_creation_scripts/download_isic.py --output-folder <path>`, **after** accepting the license on both the ISIC2019 and HAM10000 dataset pages | FLamby fed_isic2019 README |
| Official baseline hyperparameters (pooled) | `NUM_CLIENTS=6`, `BATCH_SIZE=64`, `LR=5e-4`, Adam, `NUM_EPOCHS_POOLED=20` | FLamby `common.py` |

Note on the label list: FLamby's `dataset.py`/`common.py` do not hardcode
`NUM_CLASSES` or a label-name list in the files fetched; the 8-class
ISIC-2019 taxonomy (MEL/NV/BCC/AK/BKL/DF/VASC/SCC) is confirmed from the
ISIC Challenge data page and cross-checked against independent published
ISIC-2019 classification papers (see `docs/literature_matrix.csv`). This
will be re-confirmed mechanically against the metadata CSV once the real
dataset is downloaded (Phase 0 checklist item).

**License-gated step — stopped, not bypassed.** Accepting the ISIC2019 and
HAM10000 dataset licenses requires a human clicking "accept" on
isic-archive.com / the Harvard Dataverse page; this cannot and will not be
done by the agent. The exact procedure is recorded in
`scripts/download_fed_isic2019_INSTRUCTIONS.md`. Until a human performs
that step and confirms, all Fed-ISIC2019 real-data experiments are marked
`PENDING (license acceptance required)` in `docs/execution_plan.md`, and
Phase 0/1 proceed on synthetic fixtures that mimic the same shapes,
6-center partition, and 8-class label space.

## 6. Fed-IXI Tiny (mandatory low-cost extension, verified 2026-08-25)

Verified from `flamby/datasets/fed_ixi/README.md`:

| Property | Verified value |
|---|---|
| Subjects | ~566 (IXI Tiny); standard IXI has ~600 |
| Centers | 3 London hospitals: Guy's Hospital (Philips 1.5T), Hammersmith Hospital (Philips 3T), Institute of Psychiatry (GE 1.5T) |
| Modality | T1-weighted MRI |
| Task | Binary brain/non-brain segmentation mask (not disease segmentation) |
| License | CC BY-SA 3.0, with required IXI attribution |
| Size | IXI Tiny ≈ 444 MB (standard IXI ≈ 27.4 GB) |
| Preprocessing | ROBEX brain extraction, NiftyReg affine registration to MNI, ITK reorientation; downsampled to small 3D volumes |

IXI subjects are healthy volunteers — this is **not** a neurological disease
dataset and is never described as one. Used only in Phase 7, after Phase 1
primary tables are complete, to test whether capability isolation
(coarse mask vs. fine voxel-level segmentation vs. site adapter) transfers
from 2D classification to 3D segmentation. No license-acceptance gate is
documented yet for IXI Tiny in the fetched README; this will be re-verified
before Phase 7 begins.

## 7. Metrics, baselines, datasets, planned experiments

Full definitions (utility, segmentation, privacy/attack, systems, and the
four capability-isolation composites ARR/UCG/RFC/CRE) are exactly as
specified in the project brief; they are not restated here to avoid drift
between two copies. The authoritative, current copy of each metric formula
lives in `medgate/metrics.py` docstrings once Phase 2 begins, with this file
cross-referencing it. Baselines, external/fusion dataset roles (ISIC 2020,
MILK10k, PAD-UFES-20, ISIC 2018), and the coarse label ontology
(melanocytic / keratinocytic / other, plus one alternative mapping for the
sensitivity analysis) are tracked per-phase in `docs/execution_plan.md`
rather than duplicated here.

## 8. Known limitations (living list — grows as phases run)

- Hardware is CPU-only with ~2.5 GiB available RAM; see §4. This bounds
  model size, batch size, and which baselines/attacks are feasible this
  session.
- Real Fed-ISIC2019 imagery is not yet downloaded (license-gated human
  step, §5). All "results" before that step are on synthetic fixtures and
  are labeled as such everywhere they appear.
- Literature verification (`docs/literature_matrix.csv`) is not yet
  complete for the full ~60-entry seed bibliography; entries are added as
  `VERIFIED` only after a primary-source check in this session, and
  `UNVERIFIED` otherwise. Unverified entries are not cited in
  `paper/main.tex`.
- The coarse taxonomy (melanocytic/keratinocytic/other) is an experimental
  ontology for this study, not a clinical access policy; AK in particular
  is clinically ambiguous (pre-malignant) and its bucket placement is
  argued, not assumed, in the sensitivity analysis (Phase 2/6).
- No formal cryptographic proof is claimed for LoREnc-style transformations
  in this project; they are described only as capability
  obfuscation/protection.

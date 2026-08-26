# Resource and scope decision

Status: living document. Captures what was actually decided, not a template
— update it when a decision changes rather than adding a new one. Last
updated 2026-08-26.

## 1. Submission target

**Not committed to a specific venue or deadline.** ICLR 2027's format (9-page
main text, standard structure, rigor bar) is used as a **scope reference** —
"how big/rigorous should this project be" — not a submission plan. No
Sept 2026 deadline applies.

Leaning, revisit once real-data results exist: **TMLR** (no hard deadline,
rewards reproducible benchmarks and honest negative results — fits a
capability-isolation benchmark paper better than a conference sprint would).
ICLR/ICML/workshop submission stays open if results end up strong and the
timing works out, but nothing here is gated on hitting a submission date.

## 2. Compute

| Source | Status | Notes |
|---|---|---|
| Local dev machine (this box) | Available now | CPU-only, 8 cores, 7.4GB RAM (~1.1GB free typically) — see `docs/hardware_report.md`. Ceiling for synthetic-fixture-scale work only; too small for CIFAR-100/BREEDS/Fed-ISIC2019 real training. |
| Grad school lab cluster | **Not yet arranged** | User just started grad school; no allocation confirmed. On the critical path for any real-dataset tier — user's action item to request access. |
| Google Colab | Free tier only | Not used recently. T4 GPU, ~12h session cap, can disconnect, no paid tier active. Good enough for a CIFAR-100-scale pilot. |

Real-data experiments (CIFAR-100 pilot onward) wait on Colab free tier and/or
lab cluster access — neither is blocked on money.

## 3. Budget

**Preference: $0.** Up to **~$50** is possible if free options prove
insufficient, but nothing gets spent without checking in first — this is a
reserve, not a plan.

What each tier buys (from the free-vs-$50 comparison given during scoping):

- **$0**: lab cluster (once access exists) + Colab free (T4, session limits) —
  enough for a real CIFAR-100 pilot and a modest BREEDS subset at reduced
  epochs.
- **~$50** (≈1–2 months Colab Pro): faster/more reliable GPU, longer
  sessions, fewer reconnects. Insurance against Colab-free disconnects or
  cluster queueing, not a new capability by itself. Held in reserve, not
  spent yet.

No paid APIs, no LLM-API dependency for core experiments (matches the
existing project constraint — the attack suite doesn't need one either).

## 4. Dataset access

Status as of this writing: **not yet sorted, no applications filed.** Friction
per dataset (assessed during scoping, unchanged since):

| Dataset | Friction | Plan |
|---|---|---|
| CIFAR-100 | None — auto-downloads | Use first; zero blockers. |
| Fed-ISIC2019 (FLamby) | Low — accept ISIC Archive terms via FLamby's setup script, no approval wait | Use for the medical tier once compute exists. |
| PAD-UFES-20 | None — open on Mendeley Data, CC BY 4.0 | Use as external-validation set; lowest-friction medical option. |
| BREEDS (needs ImageNet) | Low-medium — HF gated `imagenet-1k` (accept terms) faster than image-net.org; ~150GB full download | Defer until CIFAR-100 pilot is done and compute (cluster/Colab) is confirmed sufficient for the extra size. |
| Derm7pt | Medium-slow — signed license request emailed to maintainers, turnaround unpredictable | Deferred; PAD-UFES-20 covers external validation for a first pass. |

## 5. Credentials / services

None provisioned yet. Expected eventual needs: a Hugging Face account (for
gated `imagenet-1k`, if BREEDS tier happens) and optionally a free-tier W&B
for experiment tracking. Neither is committed yet. No secrets are to be
pasted into chat; use environment variables / an ignored `.env` when these
are actually set up.

## 6. Open action items (not mine to do)

- Request grad lab cluster allocation (advisor/department).
- Decide whether to file the Fed-ISIC2019/FLamby and PAD-UFES-20 requests
  now (both low-friction) or wait until compute is confirmed.

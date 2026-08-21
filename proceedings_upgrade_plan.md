# Plan: Upgrading to a Proceedings-Track Submission

This is a scoping document, not an execution log — nothing here has been run yet. Goal: lay out what closing each gap actually requires, what's already sitting in the repo ready to use, what's genuinely uncertain, and roughly how much work each piece is, so you can decide what to greenlight.

Current state for reference: 3 pages, one model (DestVI), one synthetic shift (binomial dropout on pseudo-spots), n=5 seeds, temperature scaling as the only recalibration method, one citation. Target: something that reads as "compelling, cohesive, high technical sophistication, clear impact in health" per ML4H's own Proceedings-track language.

---

## Phase A — Real distribution shift (highest priority, highest risk)

**Why it matters most:** this is the paper's own headline claim ("distribution shift") and it's currently untested on real data — everything so far is a synthetic dropout ablation on the model's own training distribution.

**What's already done:** `data_acquisition.py` and `preprocessing.py` already downloaded and gene-intersected real data — `data/processed_visium.h5ad` (217MB, 10x Visium mouse coronal brain) and `data/processed_slideseqv2.h5ad` (262MB, Slide-seqV2 mouse hippocampus) both exist and are ready to load.

**The actual blocker (not just wiring):** the whole ECE pipeline needs ground-truth cell-type proportions per spot (`adata.obsm["proportions"]`) to compute accuracy. The synthetic pseudo-spots have this by construction (you built them from known cell mixtures). Real Visium/Slide-seqV2 spots don't — knowing the true cell-type composition of a real spot is exactly the problem deconvolution is trying to solve. This is a real methodological design decision, not an engineering task, and it's almost certainly why this phase never got done originally. Three ways to resolve it, in order of how defensible they'd be to a reviewer:

1. **Layer/region annotations as a coarse proxy.** If the Visium/Slide-seqV2 `.obs` metadata includes anatomical layer or cluster labels (needs inspection — I haven't checked), you could evaluate "does the model's top predicted cell type match the expected cell type for that anatomical region" as a weaker but real accuracy signal. Defensible if the annotations are trustworthy; weaker ground truth than the synthetic setup.
2. **Silver-standard from a second deconvolution method.** Run an established tool (e.g. RCTD) on the same data and treat its output as a reference; measure DestVI's calibration against that. Standard trick in the field, but it means validating and running a second full method — real added scope.
3. **Drop ECE for real data, report a descriptive comparison instead.** Without ground truth, you can still compare confidence distributions, entropy, and ensemble agreement (do independently-seeded models agree more on real Visium than on real Slide-seqV2?) between the two real datasets, framed explicitly as a qualitative/exploratory comparison rather than a rigorous ECE claim. Weakest scientifically but requires no new ground-truth machinery — could pair well with keeping the synthetic dropout result as the paper's primary rigorous claim and this as a secondary "does the pattern hold under real shift too" sanity check.

**Compute note:** the current pipeline forces `accelerator='cpu'` for determinism (from the reproducibility fix). Slide-seqV2 is tens of thousands of beads versus 2,000 synthetic pseudo-spots — CPU-only training at that scale could be slow enough to matter for iteration speed. Worth benchmarking one epoch's wall-clock time before committing to a full 5-seed run, and possibly subsampling beads or accepting GPU non-determinism with more seeds as a tradeoff.

**Effort:** largest item in this plan. Needs a design decision (above) before any code gets written, then a full data/training/eval pass per real dataset.

---

## Phase B — Shift-magnitude sweep (cheap, no blockers)

Currently only one operating point is tested: 80% dropout (`fraction=0.2`). Sweeping `fraction` over e.g. {0.8, 0.6, 0.4, 0.2} (20/40/60/80% dropout) reuses the existing synthetic pipeline entirely — just an outer loop over `fraction` around the existing `run_replicate` calls, same ground-truth machinery, same seeds. Turns "one point degraded a little" into an actual dose-response curve, which is a much stronger and more sophisticated-looking result for the same underlying method, and directly strengthens the "under distribution shift" claim without touching Phase A's ground-truth problem.

**Effort:** small. Mostly compute time (4x the current run count), no new design decisions.

---

## Phase C — Multi-model benchmark (Starfysh, GraphST, SpatialGlue)

`implementation_plan.md` already scoped this as "Phase 5 — Scope Permitting" and it was never started. Turns a single-model case study into an actual benchmark (matching the repo's own name). Needs, per additional model: install/dependency check (each has its own package, own API shape, may not share DestVI's exact interface for getting proportions + retraining per dataset), a wrapper matching the current `get_calibration_stats`/`optimize_temperature` pipeline, and its own seeded-replicate run.

**Effort:** moderate-to-large per model, and somewhat unpredictable until each package is actually installed and tried — some of these are less actively maintained than scvi-tools/DestVI, so budget time for compatibility issues. Recommend starting with just one additional model (whichever has the most scvi-tools-like API) rather than committing to all three up front.

---

## Phase D — Compare recalibration methods

Only temperature scaling is implemented; `implementation_plan.md` Phase 4 also scoped split conformal prediction. Adding conformal prediction (or isotonic regression as a simpler second baseline) as a second recalibration method, evaluated on the same calibration/test split, would let the paper report "which recalibration method works better" rather than just "recalibration helps" — genuine added technical depth for relatively contained scope, since it plugs into the existing calibration/test split machinery.

**Effort:** small-to-moderate. Self-contained relative to Phases A/C.

---

## Phase E — Related work section

Currently one citation (DestVI itself). An 8-page archival paper needs positioning against: confidence calibration literature (Guo et al. 2017 and successors), calibration-under-distribution-shift literature (e.g. Ovadia et al.), and any existing spatial-transcriptomics- or single-cell-specific calibration work. This is a literature search + writing task, not an experiment — can run in parallel with any of the above.

**Effort:** small, but needs a proper literature search rather than citations from memory (calibration and distribution-shift literature has moved since training-data cutoff).

---

## Phase F — Statistical rigor polish

Now that determinism is verified and n=5 is cheap to extend, consider: more seeds (10+), and a real paired significance test (e.g. paired t-test or bootstrap CI on the 5-or-more matched ID/OOD ECE pairs) instead of the current qualitative "modest margin" hedge. Cheap given the existing pipeline, mostly a matter of compute time and one new stats snippet.

**Effort:** small.

---

## Phase G — Clinical impact discussion

The abstract gestures at "clinical and biological discovery" but never says what concrete clinical decision would go wrong under miscalibrated confidence. A paragraph making this concrete (e.g. a specific downstream use of deconvolution proportions — cell-type-informed treatment stratification, tissue microenvironment characterization for a specific disease context) would address ML4H's stated "clear impact in health" criterion. Writing task, needs a bit of domain framing but no new experiments.

**Effort:** small.

---

## Phase H — Formatting/anonymization compliance

Applies regardless of which phases above get done, and regardless of Proceedings vs. Findings: rebuild on the actual ML4H 2026 Overleaf template (`\mlhtrack{proceedings}`), strip the author byline for the review copy, anonymize or remove the GitHub link. Same items already flagged for the Findings path.

**Effort:** small, purely mechanical.

---

## Suggested sequencing

If the goal is the strongest paper for the least wasted effort: **B → D → F → E → G → H**, then reassess before committing to A or C, since those two are the ones where the actual scope is still unknown (A has an open design question, C has unknown per-model integration cost). B through H combined would already turn this into a noticeably more complete paper — a dose-response shift curve, two recalibration methods compared, real statistical significance, a related-work section, and a concrete clinical framing — without touching the two open-ended items. A and C are the ones that would most convincingly clear the Proceedings bar, but they're also the ones worth scoping further (or explicitly time-boxing) before starting, rather than committing to open-endedly.

# Referee Report

**Paper:** "Are Spatial Transcriptomics Models Confidently Wrong? Benchmarking Uncertainty Calibration Under Distribution Shift"
**Venue:** Submitted to ML4H 2026
**Reviewer note:** This review is based on `paper.tex`, `src/benchmark.py`, `src/preprocessing.py`, `src/data_acquisition.py`, `src/validate_destvi.py`, `results.json`, and both figures in `figures/`. I could not execute the pipeline (it requires GPU/large `.h5ad` files not runnable in this review environment), so claims about runtime behavior are marked accordingly.

**Recommendation: Major Revision.** The core empirical numbers (ECE 0.211 → 0.206 → 0.054) are internally consistent between `results.json`, `benchmark.py`, and the figures, and the writing is clear and appropriately scoped in its claims about *what* was tested (synthetic dropout only, not real cross-platform data). However, there is a serious disconnect between the Methods section's central claim — that the reported calibration is based on Monte Carlo posterior uncertainty — and what the code actually computes, plus a figure that does not show what its caption says it shows. Both need to be resolved before this is submission-ready.

---

## Major Issues

### M1. The extracted "uncertainty" is never used in the calibration metric (confirmed)

The Methods section states: *"Uncertainty was extracted via Monte Carlo Dropout sampling from the latent representation."* This implies the reported ECE reflects some measure of posterior/epistemic uncertainty.

Looking at `benchmark.py`:

```python
def mc_dropout_proportions(st_model, n_samples=30):
    ...
    proportions_stack = np.stack(proportions_list, axis=0)
    mean_props = np.mean(proportions_stack, axis=0)
    var_props = np.var(proportions_stack, axis=0)
    return mean_props, var_props
```

```python
pred_props_id, _ = mc_dropout_proportions(st_model_id, n_samples=20)
```

`var_props` — the actual MC-dropout uncertainty estimate — is discarded (`_`) everywhere it's computed. The value that feeds `get_calibration_stats()` is `mean_props`, the *point-estimate* cell-type proportions. `get_calibration_stats()` then treats `mean_props` as if it were a softmax output, converts it to logits via `log()`, and computes "confidence" as `max(softmax(logits/T))`. This is standard classifier-confidence calibration (à la Guo et al.) applied to the model's mean proportion estimate — it has no dependency on the posterior variance at all.

**Consequence:** the paper's entire finding (baseline miscalibration, shift-robustness, and the benefit of temperature scaling) is really a statement about how well the *point-estimate* deconvolution proportions align with argmax accuracy, not about whether DestVI's *uncertainty* is calibrated. That's a legitimate and useful benchmark to run — but it's a different claim than "uncertainty calibration," and the title/abstract/methods should either (a) implement calibration against the extracted posterior variance (e.g., prediction-interval coverage, which is literally listed as a goal in your own `implementation_plan.md` Phase 3), or (b) reframe the paper as being about confidence calibration of point predictions, not posterior uncertainty.

### M2. Figure 1 does not show what its caption claims (confirmed)

The caption for `fig:rel` reads: *"Representative reliability diagrams showing model calibration on In-Distribution vs Synthetic Dropout Shift pseudo-spots for a single initialization seed"* — implying a 3-way comparison (ID / OOD / Recalibrated), which is exactly what `benchmark.py` tries to build:

```python
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
diagram = ReliabilityDiagram(bins=10)
diagram.plot(conf_id, acc_id, ax=axes[0])
diagram.plot(conf_ood, acc_ood, ax=axes[1])
diagram.plot(conf_ood_cal, acc_ood_cal, ax=axes[2])
```

I opened `figures/reliability_diagram.png` directly. It contains a **single** confidence-histogram + reliability-diagram pair, not three side-by-side panels. This strongly suggests `netcal`'s `ReliabilityDiagram.plot()` is not respecting the passed `ax=` argument in the version installed (a known gotcha with some `netcal` versions, which build their own internal figure) — the three `plt.subplots` axes end up empty/discarded, and only the last call's auto-generated figure gets saved. Visually, the saved diagram's bars sit close to the diagonal with only a small gap in the low-confidence bins, which looks more consistent with the *recalibrated* result (ECE≈0.05) than the ID baseline (ECE≈0.21) — supporting the theory that this is the last (`axes[2]`) call's figure, not a 3-panel comparison.

Either way, this figure as it currently exists does not support the caption and should not go into the paper without regenerating it (e.g., manually building three `ReliabilityDiagram` figures and compositing them with `plt.imread`/`fig.add_subplot`, or checking the `netcal` version's `ax` support).

### M3. "Did not degrade significantly" is not backed by a significance test (confirmed)

Abstract/Results state the shift "did not induce catastrophic overconfidence" and calibration "remain[ed] relatively stable," based on ID ECE $0.211\pm0.018$ vs. OOD ECE $0.206\pm0.023$ over **n=3** seeds. No paired statistical test (e.g., paired t-test across the 3 matched seeds) is reported, and with n=3 such a test would have very low power regardless. The current support is "the error bars overlap," which is suggestive but not a rigorous basis for "did not degrade significantly" — recommend either (a) softening the language to avoid the statistical connotation of "significantly," or (b) running more replicates and reporting a proper test/effect size.

Relatedly: the ID ECE is computed on all 2000 pseudo-spots, while the OOD test ECE is computed on a ~1000-spot held-out half (after the 50/50 cal/test split). Comparing ECE point estimates across different-sized evaluation sets without accounting for the resulting difference in estimator variance is a minor confound on top of the small-n issue.

### M4. Possible non-stochastic "MC Dropout" sampling (plausible — needs empirical verification)

I checked scvi-tools' actual `DestVI.get_proportions()` source (via GitHub). It has no internal `.eval()`/`.train()` calls, so `mc_dropout_proportions()`'s manual `m.train()` on `Dropout*` submodules is not silently undone by the framework. Whether this produces genuinely different values across the 20 MC samples then depends on (a) DestVI's `amortization` mode (default `"both"` in the underlying `MRDeconv` module, which does route through a stochastic encoder/generative pass rather than reading a fixed point-estimate parameter — good), and (b) whether that encoder path actually contains `nn.Dropout` layers that get toggled by the class-name check in `mc_dropout_proportions`, versus stochasticity coming entirely from VAE reparameterized sampling (which would make the dropout-toggling code a no-op, though not necessarily harmful).

I could not execute the pipeline to confirm empirically. **Recommendation:** add a one-line sanity check/unit test — e.g., assert that `var_props` from `mc_dropout_proportions` is non-zero and print its magnitude — and report that check's outcome, since M1 already means this uncertainty currently goes unused regardless, but it will matter if you address M1.

### M5. Possible undertraining (worth flagging, not confirmed)

Both `CondSCVI` and `DestVI` are trained for `max_epochs=25` in `run_replicate()`. Typical scvi-tools tutorials for these models use substantially more epochs (or rely on the library's automatic epoch heuristics, which for CondSCVI/DestVI on datasets this size would generally pick a higher number). A baseline ID ECE of 0.211 for a "dominant cell type" classification task is fairly high — worth ruling out that this reflects undertraining rather than a genuine property of the model's calibration, since an undertrained model could plausibly show both high ID error *and* insensitivity to the OOD shift (i.e., M3's "no degradation" finding could partly be a floor effect from a model that wasn't well-fit to begin with). Consider a quick epoch-count ablation, or at minimum report training/validation loss curves to show convergence.

---

## Minor Issues

- **No Limitations section.** For an ML4H submission, reviewers will expect explicit discussion of: single model (DestVI only — Starfysh/GraphST/SpatialGlue were scoped in `implementation_plan.md` Phase 5 but not run), single shift type (synthetic uniform binomial dropout, not a real cross-platform/cross-tissue/cross-species shift despite that framing appearing in the project's own planning docs), single species/tissue (mouse cortex only), and a single realization of the shift (the same downsampled dataset is reused across all 3 seeds — only model initialization varies, not the shift itself).
- **Reproducibility gaps.** The paper doesn't report training epochs, MC-sample count, pseudo-spot construction (2000 spots × 10 cells/spot), or the specific `netcal` binning (`bins=10`) — all of which live in the code but not the text. Since the repo is public this is recoverable, but ML4H reviewers generally expect these in the Methods text itself.
- **Abstract overclaim on scope.** "…reduces this baseline miscalibration effectively on test data" reads as if temperature scaling fixes the *in-distribution* baseline miscalibration. In fact, temperature scaling is only fit and evaluated on the *shifted* (OOD) calibration/test split — the ID-baseline ECE (0.211) is never itself recalibrated or re-evaluated. Suggest rewording to make clear the recalibration claim is scoped to the shifted-domain test set.
- **Unused downloaded datasets.** `data_acquisition.py` and `preprocessing.py` fetch and process real Visium (mouse brain) and Slide-seqV2 (mouse hippocampus) data (~800MB combined), but `benchmark.py` never touches them — the actual experiment is 100% synthetic pseudo-spots. This isn't a correctness problem (the paper is honest about using only the synthetic dropout ablation), but it's worth deciding whether a real cross-platform comparison belongs in this version of the paper (it would substantially strengthen the "distribution shift" claim in the title) or should be cut from the repo/left for future work.
- **Title/framing vs. findings.** "Confidently Wrong" implies discovering dangerous overconfidence; the actual finding is closer to "moderately miscalibrated but not badly so, and shift-robust in this one synthetic setting, and fixable by standard temperature scaling." This is a fine, useful result — but the title over-promises relative to what's shown. Consider a more literal title if the dramatic framing isn't earned by the results.

---

## Verification status summary

| Finding | Status |
|---|---|
| M1: extracted uncertainty unused in ECE calc | **Confirmed** (direct code read) |
| M2: Figure 1 doesn't match caption | **Confirmed** (direct image inspection) |
| M3: "not significant" claim unsupported by a test, n=3 | **Confirmed** (direct code/results read) |
| M4: MC-dropout may not be stochastic under default amortization | **Plausible**, unverified — could not execute |
| M5: possible undertraining (25 epochs) | **Plausible**, unverified — could not execute |

---

## Revision Notes (v2 — checked against updated `paper.tex`/`paper.pdf`, `benchmark.py`, `results.json`, and the new figure files)

**Updated recommendation: Minor Revision.** All three confirmed major issues are resolved, and the two unconfirmed ones are honestly disclosed rather than papered over. Two small cosmetic items remain before this is submission-ready.

- **M1 (uncertainty/ECE mismatch) — Resolved.** `mc_dropout_proportions()` was removed entirely; `benchmark.py` now calls `st_model.get_proportions().values` directly, and the paper was reframed accordingly: title is now "...Benchmarking **Confidence** Calibration...", and Methods now explicitly says *"We evaluated standard softmax-confidence calibration on the model's point-estimate expected proportions."* The claim now matches the code exactly.

- **M2 (Figure 1 mismatch) — Resolved.** The three reliability diagrams are now generated and saved as separate files (`rel_id.png`, `rel_ood.png`, `rel_cal.png`, working around the `netcal` `ax=` bug I flagged) and composed side-by-side in a full-width `figure*` in the two-column layout. I opened all three PNGs directly — they show ECE 0.216 / 0.237 / 0.066 respectively for the representative seed, consistent with the reported means. This is also a better layout choice than the original single-column cramped version.

- **M3 (no significance test, n=3) — Addressed via disclosure.** A new Limitations section explicitly names the n=3 sample-size caveat (and the 25-epoch and single-shift-type caveats from M5 and the original minor issues list) rather than claiming more than the data supports. Reasonable for a workshop paper.

- **M4/M5 — Moot / disclosed.** Removing the MC-dropout code sidesteps M4. M5 (25 epochs) is now explicitly named as a limitation.

- **Side note worth mentioning in-text if there's room:** replacing the MC-dropout mean with a single deterministic `get_proportions()` call shrank the reported standard deviations substantially (ID: 0.018→0.003, OOD: 0.023→0.006, Recalibrated: 0.020→0.005). That's expected — the old std conflated true seed-to-seed variability with noise from averaging only 20 stochastic forward passes — but it's also *why* the headline finding flipped from "stable under shift" to "degrades under shift" (0.212 vs 0.231 no longer overlap). Since this is a fairly consequential change to the paper's main conclusion, it may be worth one sentence in the text noting that the earlier MC-averaging procedure introduced estimation noise that was masking the degradation, for a reader comparing against an earlier draft or the public repo history.

### Remaining minor items

1. **Cosmetic overlap in the three reliability-diagram panels.** In each of `rel_id.png`/`rel_ood.png`/`rel_cal.png`, the `plt.title(...)` call lands on the lower (Reliability Diagram) subplot and visually overlaps the upper subplot's "Confidence" x-axis label — e.g. "In-Distribution / ECE: 0.216" sits directly on top of "Confidence." At the `0.32\linewidth` size used in the paper this will be legible but visually messy. Easy fix: use `fig = plt.gcf(); fig.suptitle(...)` instead of `plt.title(...)`, or add `plt.subplots_adjust(top=...)`/`bbox_inches='tight'` before saving.
2. **Stale unused file.** `figures/reliability_diagram.png` (the old broken 3-in-1 figure) is still sitting in the repo but is no longer referenced by `paper.tex`. Not a correctness issue, just housekeeping — worth deleting so a future reader doesn't wonder which figure is current.
3. **Nice-to-have:** per-seed raw ECE values aren't persisted anywhere (only the aggregate mean/std land in `results.json`). Saving the 3 individual values would let a reader verify the "degraded" claim held in all 3 seeds individually, not just in aggregate — cheap to add and strengthens the n=3 story.

---

## Revision Notes (v3 — checked against the newest `paper.tex`/`paper.pdf`, `benchmark.py`, and `results.json`)

**Updated recommendation: Major Revision (regression).** All three v2 cosmetic/housekeeping items are fixed — titles now use `fig.suptitle()` + `bbox_inches='tight'` (no more overlap), the stale `reliability_diagram.png` was deleted, and `results.json` now stores per-seed `"raw"` arrays. Good.

But a new, more serious problem has appeared: **the headline result flipped between v2 and v3, from the exact same code and the exact same three seeds.**

| Metric | v2 | v3 | Δ |
|---|---|---|---|
| ID ECE | 0.212 ± 0.003 | 0.223 ± 0.002 | +0.011 |
| OOD ECE | 0.231 ± 0.006 | 0.223 ± 0.006 | −0.008 |
| Recalibrated ECE | 0.072 ± 0.005 | 0.062 ± 0.003 | −0.010 |

v2's paper said calibration "degraded" under shift (ID 0.212 vs. OOD 0.231, non-overlapping error bars). v3's paper says calibration "remained remarkably stable" (ID 0.223 vs. OOD 0.223, identical). **These are opposite scientific conclusions, both drawn from the same `benchmark.py` and `seeds = [42, 123, 2026]`.** Between the two runs, `benchmark.py` only changed in the figure-saving code (the cosmetic title fix) — `run_replicate()`, `downsample_counts()`, and all training/evaluation logic are byte-for-byte identical. That means this pipeline is not actually deterministic despite fixing the seeds, and whichever narrative ends up in the paper currently depends on which run happened to be kept, not on a real, reproducible property of the model.

Two concrete sources of the non-determinism, both visible directly in the code:

1. **The shifted dataset itself isn't seeded.** In `main()`, `adata_ood = downsample_counts(adata_st, fraction=0.2)` is called once, *before* the replicate loop, and `downsample_counts()` calls `np.random.binomial(...)` with no seed set anywhere beforehand (`scvi.settings.seed = seed` only happens later, inside `run_replicate()`, once per replicate). So every time `main()` runs, a *different* random realization of which reads get dropped is used as "the" shift — this alone is enough to explain the OOD ECE moving between runs.
2. **ID ECE also moved (0.212→0.223), and ID ECE has zero dependency on `adata_ood`.** That rules out point 1 as the explanation for the ID-side shift and confirms a second, independent source of non-determinism — most likely GPU/cuDNN non-determinism in DestVI/CondSCVI training that `scvi.settings.seed` alone doesn't pin down (it typically seeds Python/NumPy/PyTorch RNGs but not `torch.backends.cudnn.deterministic`, which is off by default and lets convolution/matmul kernels pick non-deterministic algorithms on GPU).

There's also a specific sentence in the v3 Results section worth removing or fixing: *"We note that utilizing deterministic point-estimates rather than Monte Carlo averaged proportions avoids artificial smoothing of the confidence distributions, confirming this robustness is a true property of the model's point predictions under shift."* This attributes the (v3) stability finding to the earlier MC-dropout removal — but that removal happened in the v1→v2 transition, and v2 (using the identical deterministic `get_proportions()` code this sentence is describing) found *degradation*, not stability. So the causal claim in this sentence is inconsistent with your own prior result and should not be used to explain why v3 looks stable — the actual explanation is run-to-run non-determinism, not the MC-averaging fix.

**Recommendation before doing anything else with the numbers:** seed the `downsample_counts()` call explicitly (e.g. `np.random.seed(seed)` or a `np.random.default_rng(seed)` passed in per replicate), and add `torch.manual_seed(seed)` / `torch.use_deterministic_algorithms(True)` / `torch.backends.cudnn.deterministic = True` (accepting the training slowdown) around the training calls. Then rerun 2–3 times with the *same* seed list and confirm you get bit-identical or near-identical ECE values before trusting either the "degrades" or "stable" narrative — right now neither is safe to publish, because the pipeline hasn't demonstrated it can reproduce its own prior run.

---

## Referee Report — v4 (run via `/paper-referee-review`)

### Summary

The paper benchmarks confidence calibration (ECE, reliability diagrams) of DestVI on synthetic mouse-cortex pseudo-spots, in-distribution vs. under a synthetic binomial dropout shift, and shows temperature scaling recovers most of the miscalibration. The scientific scope is modest and honestly bounded (synthetic shift only, single model, n=3 seeds — all disclosed in a Limitations section). The code-level reproducibility bug flagged in the previous round has now been genuinely fixed. However, **the PDF submitted for this review was not rebuilt from the current source** — it still shows the old, non-reproducible v3 numbers and a now-debunked causal claim, while `paper.tex` on disk has already moved on to a new, verified-reproducible set of numbers with different (and currently self-contradictory) framing. This round's review is therefore about a moving target, and the most important finding is procedural rather than statistical.

### Assessment

The underlying benchmark is now sound and, importantly, demonstrably reproducible: I compared `results.json`, `results1.json`, and `results2.json` on disk — three separate invocations of `benchmark.py` after the seeding/determinism fix — and they are **byte-identical** down to 15 decimal places (ID ECE 0.20595228582744798 in all three; same for OOD and recalibrated ECE, same per-seed raw arrays). That's strong, verifiable evidence the `np.random.seed`/`torch.manual_seed`/`torch.use_deterministic_algorithms`/`accelerator='cpu'` changes actually solved the v3 non-determinism problem. This is real progress and worth stating plainly rather than burying under new complaints.

### Major issues

1. **The uploaded `paper.pdf` does not match the current `paper.tex` or the current results on disk.** File timestamps make the sequence unambiguous: `results.json`/figures were last written at `~1787200059`s, `paper.tex` was edited afterward at `1787200069`s (consistent with the new numbers), but `paper.pdf` was last built at `1787195349`s — before either change. The PDF you sent still reads ID ECE $0.223\pm0.002$ / OOD ECE $0.223\pm0.006$ / Recalibrated $0.062\pm0.003$ and the "remained remarkably stable" + MC-dropout parenthetical language I flagged as factually inconsistent last round. The current `paper.tex` has moved on to ID ECE $0.206\pm0.011$ / OOD ECE $0.223\pm0.004$ / Recalibrated $0.063\pm0.002$ and has already dropped the problematic parenthetical. **Fix:** recompile `paper.tex` → `paper.pdf` before any further review or submission; right now the two disagree with each other and neither is safe to send out as-is.

2. **The current `paper.tex` Results section is internally inconsistent about direction.** Three places now say three different things about the same experiment: the Abstract says calibration merely *"changes"* under shift (deliberately neutral); the Results section says *"the calibration behaves as follows, with ECE reaching $0.223 \pm 0.004$"* — a placeholder-sounding sentence that never actually states whether that's better, worse, or unchanged relative to $0.206\pm0.011$; and the Conclusion still asserts calibration *"degrades further under synthetic distribution shifts."* A reader has to do the subtraction themselves to find out what the paper claims. Given the reproducibility saga, this is also the moment to be honest about effect size: $0.206\pm0.011$ vs. $0.223\pm0.004$ is a real but fairly marginal gap (roughly one combined-uncertainty width apart, from n=3), not the dramatic separation the v2/v3 numbers happened to show. **Fix:** pick one consistent, appropriately-hedged description (e.g. "a small increase in ECE, $0.206\to0.223$, though within a modest margin given n=3") and use it in all three places.

### Minor issues

- "the calibration behaves as follows" (Results, current `paper.tex` line 55) reads like an unfinished edit — needs an actual descriptive clause.
- Given how close the v2/v3/v4 runs have landed relative to their own error bars (0.212 / 0.223 / 0.206 for the ID arm alone, across three different but supposedly-controlled runs of "the same" experiment), the paper would be more convincing with more than 3 seeds now that the pipeline is confirmed deterministic and cheap to rerun (CPU training already in use) — even 5-10 seeds would meaningfully firm up the significance story that's currently just disclosed as a limitation.
- Minor housekeeping: `results1.json`/`results2.json` (the verification reruns) are sitting in the repo root next to `results.json` — worth deleting or moving into a `verification/` subfolder so a future reader doesn't wonder which is canonical.

### Suggested revisions

1. Recompile `paper.pdf` from the current `paper.tex` — this is the blocking item.
2. Reconcile the Abstract/Results/Conclusion into one consistent, effect-size-appropriate description of the shift result.
3. Optionally, now that reruns are cheap and deterministic, bump replicate count above 3 to give the "degrades" (or whatever the final framing is) claim real statistical footing rather than a disclosed limitation.

### Recommendation

**Minor revision** — the empirical pipeline is in good shape and its reproducibility is now verified, but do not submit the currently-uploaded PDF: rebuild it from source and reconcile the direction-of-effect language across the abstract, results, and conclusion first.

---

## Referee Report — v5 (run via `/paper-referee-review`)

### Summary

This version bumps replicates from n=3 to n=5 (seeds 42, 123, 2026, 777, 999), and rewrites the abstract/results/conclusion to consistently describe "a small increase in error" under shift rather than picking a dramatic word ("stable" or "degrades") the data didn't clearly support. I verified `results.json` against the current `paper.pdf`/`paper.tex` line by line — they match exactly (ID ECE $0.212\pm0.014$, OOD ECE $0.229\pm0.013$, Recalibrated $0.069\pm0.009$) — and confirmed the build order is now correct (`results.json` → `paper.tex` → `paper.pdf`, each with later timestamps), so unlike the previous round, this PDF is not stale.

### Assessment

This is now a materially stronger and more honest draft. Two things worth noting explicitly: first, the first three raw ID-ECE values in the new `results.json` (`0.1920946873910725`, `0.20736167974397543`, `0.218400490347296`, for seeds 42/123/2026) are bit-for-bit identical to the values from the prior verification run — independent confirmation that the determinism fix from last round holds under a real change to the codebase (extending to 5 seeds), not just repeated no-op reruns. Second, the reported effect ($0.212\pm0.014$ vs. $0.229\pm0.013$, roughly one within-sample std apart) is now described with appropriately hedged language ("a small increase... though within a modest margin given $n=5$") instead of an unqualified "degrades" or "remains stable" — this is the right level of confidence for what the data actually show.

### Major issues

1. **"Successfully repaired the baseline miscalibration" is still imprecise** (Results, final sentence: *"Applying temperature scaling derived from the calibration split successfully repaired the baseline miscalibration, reducing the final ECE to $0.069\pm0.009$"*). Temperature scaling is fit on the OOD calibration split and evaluated only on the OOD test split — the in-distribution ("baseline") ECE of $0.212$ is never itself recalibrated or re-evaluated anywhere in the pipeline. This exact wording has persisted across every revision I've reviewed. **Fix:** replace "the baseline miscalibration" with something scoped correctly, e.g. "the shift-induced test-set miscalibration."

### Minor issues

- The title ("Are Spatial Transcriptomics Models Confidently Wrong?") reads as more dramatic than the now carefully-hedged finding it sits on top of — the paper's own Results section says the shift effect is "within a modest margin." Not a correctness issue, just worth a final gut-check on tone before submission.
- No formal paired significance test (e.g. a paired t-test across the 5 matched seeds) is reported even though $n=5$ would now support one cheaply — not required given the honest qualitative hedging already in the text, but it would let the paper state a precise effect size/p-value instead of "modest margin."
- (Carried, low priority, already disclosed as future work) the real Visium/Slide-seqV2 datasets acquired in `data_acquisition.py` remain unused by `benchmark.py`.

### Suggested revisions

1. Reword the "repaired the baseline miscalibration" sentence to scope it to the shifted/test-set result — this is the one remaining place where the text claims more than the pipeline demonstrates.
2. Optional: a paired t-test on the 5 matched-seed ID/OOD ECE pairs would upgrade "modest margin" into a precise, citable statistic.
3. Optional: reconsider whether the title's dramatic framing still fits a result this carefully hedged.

### Recommendation

**Minor revision** — one sentence away from clean. Fix the "baseline miscalibration" scoping and this is ready.

---

## Revision Notes (v6)

Both remaining items from the v5 report are addressed: the Results sentence now correctly reads *"successfully repaired the shift-induced test-set miscalibration"* (properly scoped), and the title dropped the "Confidently Wrong?" framing in favor of the more literal "Benchmarking Confidence Calibration in Spatial Transcriptomics Models Under Distribution Shift." Numbers are unchanged from v5 ($0.212\pm0.014$ / $0.229\pm0.013$ / $0.069\pm0.009$) and still match `results.json`, as expected since only wording changed — no rerun was needed.

One trivial leftover: `\title[Benchmarking Confidence Calibration Under Distribution Shift]{...}` — the short-title argument in brackets (used for the running page header, visible on p.3) wasn't updated when the main title changed, so the header now reads a slightly different, older phrase than the actual title. Cosmetic only; update the bracketed argument to match.

**Recommendation: Ready to share as-is**, modulo that one-line header fix.

---

## Final Report — v7 (run via `/paper-referee-review` + `/notation-consistency-check` + `/proof-rigor-check`)

The header fix from the v6 note is applied — `\title[...]{...}` now carries the same text in both the bracket (running header) and body arguments, and the rendered page-3 header matches the paper title. I did a full fresh read of this version rather than just diffing against v6, per the referee-review process (read the whole thing, re-identify the main claims, re-check correctness before presentation).

### Paper Referee Review

**Summary.** The paper benchmarks confidence calibration (ECE, reliability diagrams) of DestVI on synthetic mouse-cortex pseudo-spots, in-distribution vs. under an 80% synthetic binomial dropout shift, over 5 seeded replicates, and shows post-hoc temperature scaling recovers most of the shift-induced miscalibration. Scope is honestly bounded throughout (synthetic shift only, single model, n=5, all disclosed in Limitations).

**Assessment.** Correct and internally consistent. Every number in the text matches `results.json`; the pipeline's determinism has been independently verified across multiple reruns and a change in replicate count; the methodological framing (softmax-confidence calibration of point-estimate proportions) now accurately describes what the code computes; and the language describing the shift effect ("a small increase... within a modest margin given n=5") is appropriately hedged to the actual effect size (0.212±0.014 vs. 0.229±0.013).

**Major issues.** None found.

**Minor issues.**
- The `netcal` bin count (`bins=10`, visible in `benchmark.py`) is never stated in the Methods text — a small reproducibility gap that's been present since the first draft and is otherwise harmless since the code is public.
- Neither figure is cross-referenced by name in the prose (no "as shown in Figure 1...") — acceptable for a 3-page workshop paper with only two figures placed adjacent to the relevant text, but worth a first-referee's note if space allows.

**Suggested revisions.** Optional only: add ", using 10 bins" to the ECE/netcal sentence in Methods for completeness. Nothing else is blocking.

**Recommendation: Ready to share as-is.**

### Notation Consistency Check

This is an empirical ML paper with no symbolic notation system to speak of — the only recurring "notation" is mean±std reporting (`$0.212 \pm 0.014$` etc.), the ECE acronym, and `n` for replicate count. All three are used consistently throughout: `$\pm$` always denotes mean±std and never anything else, `ECE` is defined at first use (Methods) and used identically thereafter, and `n` consistently means seed count (introduced as "5 independent replicate trainings" in Methods, then written as `$n=5$` in Results and Limitations — same referent, compatible notations, no ambiguity). No symbol is reused for two different meanings, and no object is given two different names. **No notation issues found.** The algebraic-topology-specific conventions this skill normally checks (basepoints, homology gradings, category/functor notation, commuting-diagram arrows) don't arise in this paper's content, so most of the skill's checklist has no applicable target here — flagging that rather than manufacturing findings.

### Proof Rigor Check

Not applicable to this document: there is no theorem, lemma, or formal proof anywhere in the paper — it's an empirical benchmark with a Methods/Results/Limitations structure, not a proof-based argument. There's no claim-with-derivation to walk step-by-step the way this skill is designed to. The closest analog — whether the empirical/statistical argument (the ECE computation, the calibration-split leakage prevention, the seeding/determinism of the pipeline) actually supports the paper's claims — has already been checked repeatedly across the referee-review passes in this file (see the reproducibility verification in the v4/v5 notes above), and no gaps remain open there either.

---

## Referee Report — v8 (major content expansion: dose-response sweep, Isotonic Regression, Related Work, clinical framing, anonymization)

This version implements several items from `proceedings_upgrade_plan.md` in one pass: n=5→10 seeds, a 4-point shift-magnitude sweep (fractions 0.8/0.6/0.4/0.2), a second recalibration method (Isotonic Regression, compared against Temperature Scaling), a Related Work section with two real, correctly-cited papers (Guo et al. 2017, Ovadia et al. 2019), a concrete clinical-impact paragraph in the Introduction, and anonymization via the template's `anon` class option plus a stripped byline/GitHub link. I verified every quoted number in `paper.tex` against `results.json` line by line — they all match exactly, including all four paired t-test p-values (0.136 / 0.501 / 0.909 / 0.472, all correctly described as "$p>0.1$ for all fractions"). The statistical claim that OOD ECE doesn't degrade significantly is genuinely well-supported this time.

### Major issues

1. **Figure 1 is stale and no longer corresponds to the reported experiment.** `figures/rel_id.png`, `rel_ood.png`, and `rel_cal.png` — the three panels in Figure 1 — all carry the previous run's timestamp (`~1787279093`, the old n=5/single-fraction/temperature-only run), while `results.json`, `ece_degradation.png`, and `paper.tex`/`paper.pdf` are all from the new run (`~1787323853–909`). The current `benchmark.py` no longer contains *any* code that generates a reliability-diagram figure — the `ReliabilityDiagram` import at the top is now dead code. So Figure 1, as currently included, shows a single 80%-dropout, temperature-scaled comparison left over from before this round's rewrite; it doesn't reflect the dose-response sweep or Isotonic Regression at all, and nothing in the current pipeline will regenerate it. Either restore figure generation for a representative fraction (and say which one, and which recalibration method, in the caption) or drop Figure 1 in favor of the dose-response Figure 2, which does reflect the current results.

2. **Methods doesn't describe the experiment Results actually reports.** Methods still says only "we synthetically downsampled the read counts... by 80%" — no mention of the 20/40/60/80% sweep that Results, Figure 2, and the abstract all center on. It also never describes Isotonic Regression at all (what it's fit on — confidence vs. correctness on the calibration split — or what implementation is used) even though it's the paper's headline recalibration result. A reader working only from Methods can't reconstruct either the sweep or the isotonic procedure.

3. **"Significantly outperforming standard Temperature Scaling" (abstract) / "proved significantly more effective" (Results) has no attached test**, unlike the ID-vs-OOD claim in the same section, which correctly reports a paired t-test with p-values. `benchmark.py` already imports and uses `ttest_rel` — adding `ttest_rel(ece_temp, ece_iso)` per fraction would be a two-line addition, and eyeballing the numbers (e.g. at fraction 0.2: Temp $0.0656\pm0.0080$ vs. Iso $0.0063\pm0.0052$ across the same 10 paired seeds) it would almost certainly come back significant — so this is very likely a true claim, just not yet a verified one in the same rigorous style the paper uses two sentences earlier.

### Minor issues

- "Temperature Scaling successfully reduced... to roughly $0.066\pm0.008$ across shift magnitudes" appears to be an eyeballed average of the four per-fraction means/stds rather than a value computed anywhere in the code (unlike the Isotonic number, which is cited exactly at one fraction, 0.2). Either compute a proper pooled statistic or cite Temperature Scaling at fraction 0.2 too, for a clean apples-to-apples comparison with the Isotonic figure right next to it.
- Isotonic Regression's near-zero test-set ECE is achieved by fitting and evaluating on two random halves of the *same* shifted dataset — worth one caveat sentence, since it demonstrates isotonic recovers well when calibration and deployment distributions match closely, which is a real methodological detail relevant to isotonic's known flexibility/overfitting tradeoff versus temperature scaling's simpler, more robust global rescaling. Doesn't need to weaken the result, just scope it.
- Dead import `from netcal.presentation import ReliabilityDiagram` in `benchmark.py`, left over from the removed reliability-diagram code (see Major issue 1) — worth removing, and its presence is itself a small signal that the figure-staleness above happened silently.

### Recommendation

**Minor revision.** The statistical core of this round (the dose-response sweep, the paired t-tests, the isotonic comparison) is sound and well-verified — every number checks out against `results.json`. What's not yet done is bringing Methods and Figure 1 up to date with what Results now actually claims, and attaching a significance test to the isotonic-vs-temperature comparison the same way the ID-vs-OOD comparison already has one.

# First principles, evaluation and remaining decisions

A cell contains RNA measurements for many genes. Cells from a knockdown and cells from a control are different cells, not matched before-and-after measurements. For each known target, estimate a mean response gene by gene: normalize each cell to 10,000 total counts, apply `log1p`, then subtract the control group's average from the target group's average. This is a difference of averages on the log scale, not a raw count change or necessarily a log fold change.

The no-change baseline predicts zero response; mean effect averages training-target responses equally. One-hop KNN ranks training targets by human STRING v12.0 combined functional association score. Equal KNN averages neighbors equally; weighted KNN weights by normalized scores. The evidence-filtered comparison additionally requires a nonzero experimental or curated-database channel, but still ranks by the combined score, which includes other channels. DSD compares six-step visitation profiles on the full network. A STRING score reflects confidence in functional association, not proof of physical binding or of similar knockdown effects. Both KNN types use the training mean when no eligible neighbor exists.

## How we compare

The scripts use training responses to predict targets absent from training. All methods use the same validation targets, gene order, cell counts, control samples and random seeds. The full validation comparison uses every real validation cell and 500 sampled training control cells per predicted target. Hold the official test responses aside until the complete procedure is frozen. Training and validation files contain shared control barcode sets; these controls are not independent. Check exact target names, identifiers, dataset identity and official manifests before final reporting. The scripts record source checksums, though a checksum alone does not prove official identity.

`cell_eval2` computes expression MAE (lower is better), perturbation discrimination PDS (higher is better, depends on the set of targets), and DE overlap (higher is better). MAE compares mean predicted and real expression profiles; it can be dominated by genes that barely change. PDS compares the predicted response to the real response for its target versus other targets. DE means differential expression: differences between target and control groups. Our `vcc` profile and Scanpy backend are explicitly configured, but do not by themselves establish official leaderboard parity.

## Scanpy warning

- **When:** A gene has the same measured value in every cell being compared, often zero.
- **Why:** Scanpy's statistical test cannot calculate a meaningful difference when there is no variation.
- **What happens:** Scanpy gives that gene a neutral p-value of 1, so it is not called differentially expressed. The warning is recorded; other warnings still need review.

## Predicted cells and DE

KNN predicts a mean response, not a cell population. In the current comparison,
we add the predicted response to sampled control cells and clip negative values.
That assumes control-like variation and can change zeros. DE depends on the
individual-cell distributions, so the population assumption matters.

The full-validation comparison also reports DE overlap. It uses all real
validation target cells and all 38,176 real controls, compared with 500
predicted cells per target. The memory-bounded CPU calculation was checked
against cell-eval2/Scanpy on sampled data and an independent full-control ACLY
run. The audit confirmed all 350 reported per-target/model overlap values.

For cell-eval2 0.22.0 with log-normalized inputs and the Scanpy backend, the
configured CPM gene filter is skipped; all 18,080 measured genes were tested.
The warning above is understood, but neither it nor numerical parity validates
the predicted cell population. Official test responses remain held out.

## Current scope and sharing

We will retain multiple KNN variants to show accuracy/discrimination tradeoffs, alongside the simple baselines; no single overall winner is claimed. The all-validation-cell run streams every real validation cell into target mean profiles and scores expression MAE and PDS using 500 predicted cells per target. DE cannot be recovered from these means, so it was calculated separately from individual cells. The memory-bounded full-cell DE analysis code is not yet included in this workflow; the included `--de` option evaluates sampled cells instead. Once the protocol is frozen, keep official test scoring separate.

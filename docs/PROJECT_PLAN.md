# VCC 2025 analysis plan

## Research question

Can a target gene's functional relationships help predict the transcriptome-wide expression response to its CRISPRi knockdown when that target's measured response was absent from training? The primary output is an expression profile, not a gene regulatory network.

## Data and splits

Use the public VCC 2025 H1 count data. Keep all cells from a perturbation target in one target-level partition. Fit on training responses, tune on validation responses, and score the test targets once after settings are frozen. If the official validation data cannot be used locally, create and document a target-level validation split inside the official training data. Store downloads in `data/raw/` and record URLs, versions, checksums, gene mappings, and sample counts.

## Shared response and prediction format

For each training target, estimate a per-expression-gene pseudobulk change from matched controls with one documented normalization rule. The mean-effect baseline averages these changes **for each expression gene across training targets**. The no-change baseline uses a control-like profile. Every method uses the same control sampling and conversion of predicted changes to nonnegative raw-count files. Fix seeds and validate a tiny file with `cell_eval2` before scoring at scale.

## Main comparison

1. **No change:** control-like response.
2. **Mean effect:** the mean training-target change for every held-out target.
3. **STRING KNN:** map targets to human STRING proteins. Start with direct links whose combined score is at least 0.4 and which have experimental or curated-database support. Rank training targets by combined score and average their response signatures with score weights. Tune the neighbor count on validation targets. Fall back to mean effect when there is no eligible neighbor, and report those targets separately. Compare against the all-channel combined-score network as a sensitivity analysis.
4. **Small MLP:** input a compact STRING-derived embedding of the target and predict a low-dimensional response signature. Fit response dimension reduction on training targets only. Use regularization and validation-based early stopping. Convert reconstructed responses to count files using the same procedure as KNN.

STRING scores represent confidence in functional association, not the magnitude or sign of a knockdown effect. The direct-edge score is the primary similarity measure. If direct-edge coverage is sparse on validation targets, compare one documented two-hop or random-walk similarity without examining test responses.

## Evaluation and leakage checks

Use the `cell_eval2` VCC 2025 profile: expression MAE, perturbation discrimination, and differential-expression overlap. Report per-target and aggregate results on identical targets for all four methods. Run a shuffled-target-label control for KNN and analyze scores by network coverage. Keep validation and test results separate.

External STRING information about a held-out target may be an input, but its measured VCC response must not enter fitting, feature selection, dimension reduction, tuning, or learned-network inference. Record the STRING release and evidence channels. Related train and test genes are expected; examine performance by similarity instead of treating relatedness itself as leakage.

## Optional RegDiffusion comparison

RegDiffusion infers a network from expression data; it does not directly predict held-out perturbation responses. After the four main methods run, it could supply an alternative network for the same KNN procedure. Infer that network only from permitted controls and training data, and keep the downstream target set and evaluation fixed. Improved response prediction would demonstrate predictive utility, not GRN correctness. A GRN accuracy claim needs its own reference and evaluation.

## Near-term handoff

- **Data audit:** obtain 2025 data, confirm counts, labels, splits, and cells per target.
- **Prediction contract:** agree on normalization, output genes, control sampling, count conversion, and a valid `cell_eval2` smoke run.
- **STRING KNN:** map identifiers, record neighbor coverage, and make validation predictions.
- **MLP:** train on training responses and predict the same validation targets.
- **Checkpoint:** produce one score table comparing all four methods, marking any unfinished method clearly.

Record owners, branches, and status in pull requests or issues rather than assigning people here.

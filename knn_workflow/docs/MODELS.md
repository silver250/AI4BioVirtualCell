# How the baseline models make a prediction

We want to predict what happens when a gene is knocked down, even when we have
not observed that knockdown in training. The models below predict an average
response for every measured gene. They differ in how they choose and combine
known training responses.

## First, prepare the training responses

For each training target, normalize each cell to 10,000 total counts and apply
`log1p`. For each measured gene, subtract the mean of the non-targeting
controls from the mean of that target's cells. The result is one response
vector per training target. It is a difference of group averages on the log
scale; individual control and target cells are not matched.

See `src/cellmates/preprocessing.py` and `scripts/prepare_training_effects.py`.

## Two simple baselines

- **`no_change`:** predict a response of zero for every gene, regardless of
  the target. The generated cells therefore start as sampled control cells.
- **`mean_effect`:** take the equal-weight average of all training-target
  response vectors. Predict that same response for each unseen target. This
  is the mean *training effect*, not the mean control-cell expression.

See `src/cellmates/baselines.py`.

## One-hop KNN

A neighbor is a training target directly connected to the unseen target in the
human STRING v12.0 functional association network. For each unseen target:

1. Match its name and the training names to unique STRING preferred names.
2. Keep edges with combined STRING score at least 0.4. Rank eligible training
   neighbors by that combined score and select up to five; equal scores are
   ordered by gene name. If fewer than five qualify, use those available.
3. Combine their training response vectors, gene by gene:
   - **`direct_equal`:** give every selected neighbor the same weight. With
     two neighbors, each contributes one half.
   - **`direct_score`:** use the same selection, but divide each neighbor's
     combined score by the sum of selected scores. Scores 0.8 and 0.4 would
     give weights of two thirds and one third, respectively.
   - **`evidence_equal`:** use the equal-weight method after additionally
     requiring the experimental or curated-database evidence channel to be
     nonzero. Selection is still ranked by the combined score. Nonzero is a
     permissive support requirement, not a separate high-confidence threshold.

The combined score reflects confidence in an association; it is not an effect
size and does not prove a physical interaction. The support channels can
include transferred evidence. If no training neighbor qualifies, each method
uses the mean training effect. Neighbor names, scores, weights and fallback
status are written to local run outputs for inspection.

See `src/cellmates/direct.py` and `scripts/prepare_networks.py`.

## DSD KNN

**Diffusion state distance (DSD)** uses the wider network to find training
targets with similar neighborhoods, even without a direct edge.

1. Use the full human STRING network with combined score at least 0.4,
   including proteins that are not training or validation targets.
2. Imagine a random walk starting at each target. Each step follows an edge
   with probability proportional to its combined score. Sum the expected visits
   at the starting point and after each of six steps.
3. Compare the unseen target's visit pattern with each training target's
   pattern by adding the absolute differences across proteins. A smaller sum
   means more similar patterns. Only targets in the same connected component
   with overlapping visit profiles are eligible.
4. **`dsd_equal`** averages up to the five closest eligible training responses
   equally. **`dsd_equal_k10`** does the same with up to ten. If none qualify,
   use the mean training effect.

These DSD baselines use the combined-score graph; the experimental/database
support requirement above applies to `evidence_equal`, not to DSD here. The
walk probabilities use STRING scores, but the selected response vectors are
averaged equally.

See `src/cellmates/dsd.py`, `scripts/prepare_networks.py`, and the
`dsd_effects` function in `scripts/run_validation.py`.

## From a predicted response to evaluation

For the reported full-validation comparison, add each predicted response to the
same 500 sampled training control cells and replace negative values with zero.
This is a simplifying assumption about cell-to-cell variation. Expression MAE
and PDS use all real validation cells through group averages. Differential
expression (DE) is calculated separately from individual cells and all real
validation controls. The same validation targets have been used for model
exploration; the official test responses remain held out. See
[assumptions and evaluation](ASSUMPTIONS.md) for metric meanings and limitations.

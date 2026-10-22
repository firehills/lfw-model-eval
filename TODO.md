# Face verification improvement backlog

These are proposed experiments, not implemented features or guaranteed accuracy
improvements. See [README.md](README.md) for the current evaluation protocol.
Prioritize lower false rejection at a fixed, independently calibrated false
accept rate, not higher raw accuracy.

## Current baseline

The evaluator normally L2-normalizes each image embedding, averages the
enrolment embeddings for each identity, and L2-normalizes the resulting
template. Each probe is compared with every identity template, with one explicit
threshold deciding acceptance. Disabling normalization changes this baseline.

Saved `embeddings.npz` files contain embeddings, templates, identity labels,
source paths and splits. Many grouping and scoring experiments can reuse these
without rerunning inference. Image-quality features, new preprocessing and
different models require additional data or inference. Saved normalized vectors
cannot recover their original embedding magnitudes.

## P0: Establish trustworthy comparisons

- [ ] Create separate development/calibration and final evaluation partitions.
  Use identity-disjoint development subjects where feasible. Fit grouping rules,
  quality weights, score transforms and thresholds without inspecting final test
  scores. Within each identity, only enrolment images may construct templates.
- [ ] Preserve the existing lexicographic enrolment/test split as the baseline.
  Any alternative sampling protocol must be explicit and reproducible; never
  silently move difficult probes into enrolment or remove them.
- [ ] Calibrate each strategy's threshold to the same target FAR on development
  data, freeze it, and report the actual FAR and TAR/FRR on held-out evaluation.
  Reusing the baseline threshold is not a fair comparison when score
  distributions change.
- [ ] Report confusion counts, comparison totals, TAR/FRR at the chosen operating
  point, low-FAR ROC behavior, and descriptive AUC/EER. Report rank-1 separately.
  Increasing a similarity threshold trades fewer false accepts for more false
  rejects; it does not improve embedding separation.
- [ ] Quantify uncertainty with subject-aware resampling rather than treating
  correlated comparisons as independent. Include paired comparisons between
  strategies on the same subjects and probes.
- [ ] Collect enough representative impostor evidence for the target FAR.
  Zero observed false accepts is not proof of zero FAR or a hard bound of
  `1 / comparisons`. Under independent trials, a rough 95% upper bound is
  `3 / comparisons`; shared subjects invalidate that simple assumption.
- [ ] Build error-analysis views linking false accepts and false rejects to
  probe images, enrolment images, scores and thresholds. Inspect alignment,
  image quality, lookalikes and potential label errors. Document suspected data
  problems instead of silently deleting hard cases.

## P1: Group multiple enrolment embeddings more effectively

For the cosine experiments below, let `e_i` be a unit enrolment embedding,
`p` a unit probe, and `unit(v)` L2-normalization. Reject nonfinite or near-zero
vectors explicitly. Each method must produce one final score per
probe/claimed-identity pair, not inflate the number of genuine trials.

| Experiment | Grouping or scoring method | Potential benefit | Main risk |
|---|---|---|---|
| Quality-weighted centroid | `unit(sum(w_i * e_i))`, with nonnegative quality weights | Reduces the influence of blur, occlusion or poor alignment | Quality estimates can be biased or discard useful pose diversity |
| Consistency-weighted centroid | Weight images by agreement with other enrolment images | Limits isolated bad embeddings without a hard exclusion | Legitimate profiles or appearance changes may look like outliers |
| Medoid template | Choose the enrolment embedding with highest average similarity to the others | Uses an observed face rather than an averaged direction | Loses complementary information from other images |
| Robust centroid | Compare spherical robust estimation, geometric median followed by normalization, or trimmed aggregation | Reduces sensitivity to contaminated enrolment | Unstable with very few images; trimming can hide difficult cases |
| Multiple cluster centroids | Cluster one person's enrolment images by angular distance; normalize one centroid per cluster | Represents distinct poses or appearances that one mean can blur | Small clusters overfit; taking the best cluster increases false-match opportunities |
| Pose/condition groups | Separate templates for frontal/profile, glasses, lighting or capture session | Keeps meaningful appearance modes distinct | Requires reliable condition metadata and enough samples per group |
| Diversity-selected exemplars | Keep representative, high-quality images with farthest-point or coverage-based selection | Avoids a gallery dominated by near-duplicate captures | Outliers may be selected as apparently diverse examples |
| Hierarchical templates | Keep a global centroid plus condition centroids or exemplars | Combines stable identity evidence with appearance coverage | More storage, scoring and calibration complexity |

- [ ] Start with the weighted centroid and medoid as inexpensive baselines.
  Keep unweighted mean pooling available and unchanged.
- [ ] Measure pairwise enrolment similarities and mean resultant length
  `norm(sum(e_i)) / n` as consistency diagnostics. Low concentration can indicate
  poor captures, mixed identities or real appearance diversity; it is not proof
  of an invalid identity.
- [ ] For quality weights, evaluate blur, face size, pose, occlusion, landmark
  reliability and exposure. Detector confidence is not automatically a reliable
  recognition-quality score. Use embedding magnitude only when the model
  explicitly supports that interpretation.
- [ ] Learn weighting and clustering settings on development data. Define the
  minimum support per cluster and behavior for one-image enrolment explicitly.
  If quality or validity requirements fail, return an explanatory enrolment
  failure or request recapture; do not silently substitute a template.
- [ ] Make exclusions explicit, retain their provenance, and report their
  counts and end-to-end impact. Never improve recognition metrics merely by
  removing difficult people.

## P1: Fuse evidence across enrolment images

Instead of scoring only a pooled template, compare a probe with each enrolment
embedding: `s_i = dot(p, e_i)` for cosine. Investigate:

- [ ] **Mean or quality-weighted mean score.** All images contribute evidence.
  For unit vectors, mean cosine is `dot(p, mean(e_i))`; unlike cosine against
  the normalized centroid, it retains the mean's concentration factor. It is
  not necessarily equivalent to the current template score.
- [ ] **Maximum score.** Accept if any enrolment image matches strongly.
  This may recover genuine matches across pose changes, but also increases
  false accepts. Calibrate against the actual number of templates.
- [ ] **Top-k mean.** Average the best few similarities to balance coverage
  against the brittleness of a single maximum.
- [ ] **Median or trimmed mean.** Reduce the effect of isolated high or low
  scores, while checking that rare legitimate appearance modes still match.
- [ ] **Soft maximum / log-mean-exp.** Smoothly interpolate between mean-like
  and max-like evidence using a development-tuned temperature. Normalize for
  template count, but do not assume this removes all enrolment-count effects.
- [ ] **Consensus voting.** Require a calibrated minimum number or fraction of
  enrolment images to pass, possibly with a stronger single-match alternative.
  This can reject accidental matches but hurt users with diverse appearances.
- [ ] **Global-plus-local fusion.** Combine centroid similarity with the best
  exemplar or cluster score using a simple calibrated weighted rule.
- [ ] **Learned score fusion.** Train a small calibrated classifier or
  likelihood-ratio model from centroid, top-k, quality and dispersion features.
  This needs adequate development subjects and controls against overfitting.

For all fusion methods, enrolment images are correlated evidence, not independent
votes or probabilities. Evaluate sensitivity to enrolment count and duplicate
images. Adapt the ordering explicitly for distance scores, where lower is better.

## P2: Calibrate decisions and model score distributions

- [ ] Compare a global threshold with regularized quality- or
  enrolment-count-conditioned thresholds. Subject-specific thresholds need
  substantially more evidence than a handful of that subject's images.
  Never derive them from that person's test probes.
- [ ] Explore within-person dispersion or uncertainty-aware matching, such as
  shrinkage covariance models or probabilistic embeddings. Full per-person
  covariance is unreliable when embedding dimension greatly exceeds image
  count; use pooled, diagonal or strongly regularized estimates.
- [ ] Evaluate cohort-based score normalization using separate background
  identities (for example Z-, T- or symmetric normalization). Measure dependence
  on cohort composition and computational cost.
- [ ] Learn centering, PCA, whitening, metric learning or PLDA-like scoring only
  on representative development/training identities. Re-normalize as required
  and validate each transform; these can damage an already well-trained model.
- [ ] Calibrate scores to likelihood ratios, or probabilities under an explicit
  deployment prior, using held-out logistic or isotonic calibration. A cosine
  score is not a probability; calibration alone does not improve ranking.
- [ ] Evaluate a three-way accept / reject / recapture decision near the
  threshold. Report the recapture or abstention rate and its operational cost;
  do not remove abstentions from the denominator to inflate accuracy.
- [ ] For 1:N identification, separately investigate best-versus-second-best
  margins and gallery-size-aware thresholds. These change the task and are not
  substitutes for claimed-identity verification.

## P2: Improve the evidence before grouping

- [ ] Collect additional enrolment images across sessions, pose, illumination,
  glasses and appearance changes. Measure diminishing returns as enrolment
  count grows; more near-duplicates are not necessarily more useful evidence.
- [ ] Verify model-specific color order, normalization, alignment landmarks,
  crop size and layout against the model's documented export contract.
  Keep these in configuration profiles, not model-name conditionals.
- [ ] Assess detector/alignment quality on real images and report failures
  separately from embedding-level false rejects. Recapture guidance may improve
  end-to-end usability more than a scoring change.
- [ ] Test model-supported horizontal-flip, multi-crop or test-time augmentation
  fusion, with correct re-normalization and measured latency. Improvements are
  not guaranteed, and training-time assumptions may not transfer.
- [ ] For video or burst capture, pool only valid frames from the same tracked
  person using quality and temporal consistency. Test track contamination,
  correlated frames, added latency and repeated-attempt security risk.
- [ ] Compare recognition backbones and input resolutions under the same
  protocol, then assess ONNX/TFLite export and float-versus-quantized accuracy.
  Record provenance, licensing, hashes and any possible training/test overlap.
- [ ] Explore model ensembles with separately calibrated score fusion. Do not
  average embedding coordinates from unrelated models: their spaces are not
  aligned. Measure added inference, storage and latency cost.
- [ ] If suitable licensed data exists, consider domain adaptation, supervised
  metric-learning fine-tuning, hard-negative mining or distillation. Use a
  training split distinct from development and evaluation subjects; check
  overfitting, demographic performance and export compatibility.

## P3: Production safeguards and broader validation

- [ ] Validate on representative cameras, lighting, age ranges, skin tones,
  occlusions and time gaps. Report subgroup and worst-condition error rates
  with uncertainty rather than relying on one aggregate LFW result.
- [ ] Add an explicitly separate unknown-identity/open-set evaluation with
  unenrolled probes and realistic gallery sizes. Current reports do not
  establish unknown-person rejection performance.
- [ ] Evaluate presentation-attack detection, replay resistance and capture
  integrity separately. Better face matching does not establish liveness.
- [ ] Bound retries and evaluate session-level false accept/reject outcomes.
  Multiple attempts are not necessarily independent; per-comparison FAR is
  not the probability that an account remains secure over many attempts.
- [ ] If templates evolve, update only after strong independent authentication
  or controlled re-enrolment. Test poisoning, appearance drift and rollback;
  never automatically learn from every accepted probe.
- [ ] Protect biometric embeddings and image links with access controls,
  retention/deletion policies and appropriate encryption. Keep reports, images
  and embeddings out of version control.

## Implementation and acceptance checklist

- [ ] Introduce explicit, versioned template and score-fusion settings, keeping
  mean pooling as the default. Keep shared logic under `face_eval/` and root
  CLIs thin.
- [ ] Record strategy, hyperparameters, calibration provenance, thresholds,
  enrolment/cluster counts, quality decisions and model/manifest hashes.
  Extend report schemas deliberately; do not silently change array meanings.
- [ ] Keep image-to-template membership and score contributions traceable so
  false accepts and rejects can be explained.
- [ ] Preserve finite/nonzero vector checks, score direction, inclusive threshold
  boundaries, no split leakage, no implicit overwrite and no CPU fallback.
- [ ] Separate embedding inference timing from grouping/scoring and end-to-end
  latency. Measure template memory and gallery scaling as well as accuracy.
- [ ] Add offline tests for one-image enrolment, repeated images, conflicting
  enrolment, multimodal identities, tied scores, invalid weights and vectors,
  exact confusion counts, schema persistence and deterministic grouping.
- [ ] Run the full pipeline/runtime validation where affected, plus real
  detector/model smoke tests for preprocessing changes and GPU-image checks
  for GPU changes. Update the README and example profiles for new options.
- [ ] Promote a strategy only after held-out evidence shows a useful TAR/FRR
  improvement at the required FAR without unacceptable subgroup regressions,
  exclusions, storage growth or latency. Report inconclusive outcomes honestly.

## Suggested first experiments

1. Establish calibrated operating points and inspect current errors.
2. Compare the existing normalized mean with a medoid and quality-weighted mean.
3. Compare centroid scoring with mean, maximum and top-k exemplar score fusion.
4. Try a small number of enrolment-only clusters when enough images exist.
5. Evaluate repeatability, uncertainty, enrolment-count effects and runtime cost
   before considering learned fusion or model training.

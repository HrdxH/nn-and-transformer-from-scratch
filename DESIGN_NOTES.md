# Design notes — Part A: Neural network from scratch

## Architecture

- 784 -> 32 -> 32 -> 10, fully connected, trained on MNIST.
- Two hidden layers of 32 neurons each (started at 16, widened to 32 after
  observing accuracy gains — see Experiments).
- Output layer: 10 neurons (one per digit class).

## Activation functions

- **ReLU** over sigmoid for hidden layers, to avoid vanishing gradients in
  deeper networks (Glorot & Bengio, 2010). Sigmoid's derivative flattens
  toward zero for large-magnitude inputs; ReLU's derivative is a constant
  1 for positive inputs, so gradients don't shrink layer over layer.
- **Softmax** for the output layer, standard for multi-class classification
  since it produces a true probability distribution (all outputs positive,
  summing to 1), unlike independent sigmoid outputs per class.
- Paired with **cross-entropy loss** (implicit in the implementation) —
  this pairing gives a clean combined gradient at the output layer,
  `dZ = A - y_onehot`, avoiding a longer chain-rule expansion.

## Weight initialization

- **He initialization** (He et al., 2015): weights scaled by
  `sqrt(2 / n_in)` rather than an arbitrary small constant.
- Needed specifically because ReLU zeroes out roughly half its inputs on
  the forward pass; without compensating for that variance loss, signal
  strength shrinks across layers even before training begins. The
  `sqrt(2/n)` factor keeps each layer's output variance roughly matched
  to its input variance at initialization.
- Biases initialized to zero — no equivalent compensation needed.

## Bug encountered: dead ReLU collapse

- **Symptom**: accuracy stuck at exactly 11.25% for hundreds of epochs —
  the network had collapsed to predicting a single class for every input.
- **Cause**: naive initialization (`* 0.01`, not scaled for ReLU) combined
  with a learning rate of 0.1 pushed many neurons' pre-activation values
  deeply negative early in training. ReLU zeroed their output; ReLU's
  derivative is also zero there, so those neurons received zero gradient
  and could never recover — permanently "dead."
- **Fix**: switched to He initialization and lowered the learning rate to
  0.01. Result: stable training, reaching ~80% test accuracy.
- **Takeaway**: initialization scheme and learning rate are not
  independent choices — an initialization mismatched to the activation
  function can make an otherwise reasonable learning rate destructive.

## Training procedure

- Standard (full-batch) gradient descent initially: one forward/backward
  pass and one weight update per epoch, using the entire training set.
- Switched to **mini-batch gradient descent** (batch size 128): weights
  updated after each batch rather than once per epoch, giving ~437
  updates per epoch instead of 1 (56,000 training images / 128).
  Data reshuffled each epoch (`np.random.permutation`) so batch
  composition varies across epochs.
- Result: 96.4% train / 95.5% test accuracy in just 50 epochs, versus
  90% train / 90% test after 1500 full-batch epochs — both faster
  convergence and a higher ceiling, attributable to the much higher
  frequency of weight updates and the regularizing effect of gradient
  noise from smaller batches.

## Momentum (tested, not adopted)

- Added momentum (exponentially weighted moving average of gradients,
  beta = 0.9) on top of mini-batch gradient descent.
- Result: 96.34% train / 95.59% test — statistically indistinguishable
  from mini-batch alone (96.37% / 95.52%).
- **Interpretation**: momentum's main benefit is smoothing noisy gradients
  and navigating awkward loss curvature. With a moderate batch size (128,
  already fairly low-noise), a shallow 3-layer network, and a
  well-behaved dataset like MNIST, there was little for momentum to
  correct for within 50 epochs. Likely to matter more with smaller batch
  sizes or deeper architectures — noted as a direction for future testing
  rather than pursued further here, to keep project scope on track.

## Train/test methodology

- 80/20 train/test split (`sklearn.train_test_split`, fixed random seed
  for reproducibility).
- Test accuracy tracked alongside train accuracy throughout to check for
  overfitting. Across all experiments, test accuracy stayed within ~1
  percentage point of train accuracy — no meaningful overfitting observed
  at this model scale.

## Results summary

| Experiment    | Config                                   | Train acc | Test acc |
|---------------|-------------------------------------------|-----------|----------|
| Baseline      | 16-16-10, naive init, lr=0.1              | 0.11      | 0.11     |
| Fix           | 16-16-10, He init, lr=0.01                | 0.81      | 0.78     |
| Wider layers  | 32-32-10, He init, lr=0.01                | 0.90      | 0.90     |
| Mini-batch    | 32-32-10, batch=128, 50 epochs            | 0.9637    | 0.9552   |
| + Momentum    | same + beta=0.9                           | 0.9634    | 0.9559   |

## Context relative to published results

- Current MNIST state-of-the-art is 99.84% (capsule network architecture).
  Fully-connected architectures similar in kind to this one have reached
  roughly 98.8-98.9% at larger scale (e.g. 800 hidden units per layer);
  >97% is generally considered a good result on this dataset.
- The gap between this project's 95.6% and state-of-the-art is primarily
  architectural rather than a tuning issue: convolutional layers exploit
  the 2D spatial structure of images (local pixel correlations) that a
  fully-connected network discards by flattening the input. Closing this
  gap further would require moving to a CNN, noted here as a natural
  extension rather than pursued in this project's scope.

## Engineering practices

- Git version control with `.gitignore` excluding `venv/`; incremental
  commits per experiment for a visible, honest progress history.
- `RESULTS.md` maintained as a running experiment log (config -> train/
  test accuracy -> notes) to preserve comparisons across code changes
  that would otherwise overwrite prior results.

  "Implemented single-head attention first to understand the core mechanism, then extended to multi-head attention (4 heads) — multi-head lets the model track multiple types of relationships in parallel (e.g., different attention heads may specialize in different positional relationships relevant to the reversal task)."

  # Design notes — Part B: Transformer from scratch, length generalization

## Research question

Does positional encoding scheme affect a from-scratch transformer's ability
to generalize to sequences longer than it was trained on, and does this
depend on how position-dependent the task itself is?

## Why this question, not text generation

Deliberately chose not to replicate a standard "train a small GPT on
Shakespeare" project. That approach has no clean, quantifiable success
metric and doesn't produce a comparable, reportable result. Instead framed
Part B around a real, currently active research question in the literature
(length generalization in Transformers), using a synthetic task with
exact-match accuracy as the metric — mirroring the ablation-study structure
used in Part A, and giving the project a genuine research contribution
rather than a reproduction of a known demo.

## Architecture

- Built from scratch on top of PyTorch's `nn.Module`/autograd (unlike Part
  A, which used raw NumPy with hand-written backprop) — implementing
  attention's backward pass by hand was judged low learning-value relative
  to time cost; autograd is standard practice for transformer work.
- Components implemented from scratch: token + positional embeddings,
  multi-head self-attention (scaled dot-product, Q/K/V projections),
  feedforward sublayer, residual connections + LayerNorm, stacked
  transformer blocks.
- Shared architecture across all experiments and encoding variants:
  d_model=32, num_heads=4, d_ff=128, num_layers=2, Adam optimizer
  (lr=0.001) — kept identical across conditions so that any accuracy
  difference is attributable specifically to the positional encoding
  scheme, not confounded by architecture size.
- Refactored shared model classes into `model_utils.py` after Experiment 1,
  so Experiment 2 (and any future extensions) import architecture code
  rather than duplicating it across notebooks. Experiment 1's original
  notebook was left with its inline definitions rather than retroactively
  refactored, since it was already complete and validated.

## Positional encoding variants implemented

- **Sinusoidal** (Vaswani et al., 2017) — fixed sine/cosine functions of
  position and embedding dimension, not learned.
- **Learned positional embeddings** — position treated as another token to
  be embedded via a trainable lookup table, same mechanism as token
  embeddings.
- **NoPE (no positional encoding)** — control condition, token embeddings
  only, no positional information supplied to the model at all.
- ALiBi and other relative schemes were considered but not implemented,
  given time constraints and that the three-way comparison above already
  produced a clear, reportable contrast — noted here as a natural
  extension for future work.

## Task design

Two synthetic tasks were used, deliberately chosen to differ in how much
they depend on token position:

- **Experiment 1 — sequence reversal**: output the input sequence in
  reverse order. Inherently positional — the correct output at each
  position depends entirely on where a token sits in the input, not just
  its identity.
- **Experiment 2 — counting**: given a sequence and a target digit, output
  how many times the target appears. Deliberately chosen as a contrast to
  reversal — the answer does not depend on token position at all
  (shuffling the input would not change the correct answer).

Using two tasks with different positional dependence, rather than one, was
a deliberate design choice to test whether conclusions about positional
encoding schemes generalize across task types or are task-specific.

## Methodology revision: single-length vs. range-based training

Experiment 1 trained on a single fixed sequence length (6) only. An early
version of Experiment 2 mirrored this setup, but verification (checking
accuracy at lengths shorter than the trained length, which should be
"easy" if generalization were general rather than length-specific) revealed
accuracy peaked sharply at exactly length 6 and degraded in both
directions — indicating the model had overfit to one specific length
rather than learning a length-invariant mechanism at all.

**Revision**: Experiment 2 was rebuilt to train on a *range* of lengths
(3-6) rather than a single fixed length, with sequence length randomized
per training batch. This isolates genuine extrapolation (performance
beyond the entire trained range) from single-length overfitting (failure
even on lengths close to, but not exactly equal to, one memorized value).
This is closer to standard practice in the length-generalization
literature, which typically trains on a length range and tests beyond its
upper bound.

This inconsistency between Experiment 1 (fixed length) and Experiment 2
(range-based, after revision) is noted explicitly as a limitation — a
consistent range-based setup across both experiments would strengthen a
follow-up version of this work, but was not retrofitted into Experiment 1
given time constraints and that Experiment 1's results remain valid and
informative as reported.

## Verification practices used

Before trusting any "it doesn't work" result as a genuine finding rather
than a bug, checked:
- Whether the model performed well at the trained length/range before
  concluding anything about lengths beyond it.
- Whether failure patterns were explainable (e.g. NoPE's inability to learn
  reversal follows directly from the task's positional dependence; the
  counting model's bias toward small predicted counts at long lengths
  matches the count distribution seen during training) rather than
  resembling arbitrary or inconsistent breakage.
- Whether results were directionally consistent with published findings on
  the same general phenomenon, without assuming the literature must be
  reproduced exactly at this small scale.

## Results summary

See `RESULTS_PART_B.md` for full tables and per-experiment analysis.
Headline finding: both *whether* positional encoding is necessary, and
*which* encoding scheme generalizes best, are task-dependent — NoPE failed
completely on reversal but partially succeeded on counting; sinusoidal and
learned PE tied on reversal but learned PE clearly outperformed sinusoidal
on counting. No scheme achieved robust generalization to substantially
longer sequences on either task.

## Related work

- Vaswani et al., "Attention Is All You Need" (2017) — original
  Transformer architecture and sinusoidal positional encoding.
- Press, Smith & Lewis, "Train Short, Test Long: Attention with Linear
  Biases Enables Input Length Extrapolation" (ICLR 2022) — introduces
  ALiBi, defines this research area.
- "The Impact of Positional Encoding on Length Generalization in
  Transformers" (NeurIPS 2023) — systematic comparison of PE schemes
  (APE, T5 Relative PE, ALiBi, RoPE, NoPE) on reasoning/math tasks; found
  NoPE outperforms explicit schemes. This project's finding that NoPE
  cannot even learn a strongly positional task (reversal) suggests their
  conclusion may not extend to tasks where position is load-bearing to the
  task definition itself, rather than incidental to it.
- Ruoss et al., "Randomized Positional Encodings Boost Length
  Generalization of Transformers" (ACL 2023) — proposes a fix for the
  out-of-distribution positional signal problem, additional context for
  future extensions of this work.

## Engineering practices

- Git version control, incremental commits per experiment/revision.
- `RESULTS_PART_B.md` maintained as the experiment log, updated after each
  major result (mirroring the practice established in Part A).
- Shared architecture code factored into `model_utils.py` to support
  multiple experiment notebooks without duplication.
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
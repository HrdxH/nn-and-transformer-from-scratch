# Part B — Length generalization experiment

**Research question:** does positional encoding scheme affect a from-scratch
transformer's ability to generalize to sequences longer than it was trained
on, and does this depend on how position-dependent the task itself is?

**Shared architecture across all runs:** d_model=32, num_heads=4, d_ff=128,
num_layers=2, Adam optimizer (lr=0.001), 1000 training steps.

---

## Experiment 1: Reversal task (inherently positional)

**Task:** given a sequence of digits (0-9), output the sequence reversed.
Exact-match accuracy per position. Trained on a single fixed length (6).

| Test length      | Sinusoidal | Learned PE | NoPE (control)         |
|-------------------|-----------|------------|-------------------------|
| 6 (trained)       | 1.000     | 1.000      | ~0.35-0.40 (never reached 1.0) |
| 8                 | 0.213     | 0.254      | not evaluated           |
| 10                | 0.166     | 0.141      | not evaluated           |
| 12                | 0.247     | 0.172      | not evaluated           |
| 15                | 0.108     | 0.137      | not evaluated           |
| 20                | 0.128     | 0.130      | not evaluated           |
| 25                | 0.153     | 0.119      | not evaluated           |

**Findings:**

- **NoPE failed to learn the base task at all**, plateauing around 35-40%
  accuracy even at the trained length. Reversal is inherently
  positional — the correct output for a token depends entirely on its
  position, not just its identity — so a model with no way to represent
  position cannot solve it, regardless of sequence length.
- **Sinusoidal and learned positional encoding performed almost
  identically**: both reached perfect accuracy at the trained length, and
  both collapsed to near-random performance (~0.11-0.25) immediately past
  the training length, with no clear advantage for either scheme.
- **Interpretation:** for this task, the meaningful dividing line was
  "having an explicit positional encoding at all" vs. "having none" — not
  "which encoding scheme." Both working schemes failed to generalize past
  the training length in essentially the same way.
- **Methodological note:** this experiment trained on a single fixed
  length (6) only. As a result, its "length generalization" failure
  conflates two effects: generalizing beyond the trained length, and
  generalizing away from one narrowly memorized length at all. Experiment
  2 (below) was redesigned to isolate these two effects — see the revision
  note there.

---

## Experiment 2: Counting task (less positional)

**Task:** given a sequence of digits (vocab_size=5) and a target digit,
output how many times the target digit appears in the sequence. This task
does not depend on token *position* — shuffling the input sequence would
not change the correct answer — making it a deliberate contrast to
Experiment 1.

**Revision note:** an initial version of this experiment trained on a
single fixed length (6), mirroring Experiment 1. Verification showed
accuracy peaked sharply at exactly length 6 and degraded in *both*
directions (shorter and longer) — indicating the model had overfit to one
specific length rather than learning a length-invariant mechanism. The
experiment was revised to train on a *range* of lengths (3-6) instead,
which isolates true extrapolation (beyond the trained range) from
single-length overfitting. All results below use this revised, range-based
training setup.

| Test length | Sinusoidal | Learned PE | NoPE |
|---|---|---|---|
| 3 (in range) | 1.000 | 1.000 | 0.935 |
| 4 (in range) | 1.000 | 1.000 | 0.955 |
| 5 (in range) | 0.995 | 1.000 | 0.820 |
| 6 (in range) | 0.970 | 0.995 | 0.600 |
| 8  | 0.690 | 0.735 | 0.215 |
| 10 | 0.170 | 0.510 | 0.080 |
| 12 | 0.070 | 0.295 | 0.095 |
| 15 | 0.030 | 0.125 | 0.020 |
| 20 | 0.025 | 0.060 | 0.040 |
| 25 | 0.005 | 0.025 | 0.010 |

**Findings:**

- **NoPE could now learn the base task**, unlike in Experiment 1 — reaching
  60-95.5% accuracy within the trained length range, confirming that
  positional encoding's necessity is task-dependent: essential for
  reversal, merely helpful (not required) for counting. Its training was
  visibly noisier than the other two schemes (fluctuating loss), and its
  in-range accuracy was consistently lower than sinusoidal or learned PE.
- **Learned PE clearly outperformed sinusoidal at every length beyond the
  trained range** (e.g. 51.0% vs 17.0% at length 10; 29.5% vs 7.0% at
  length 12) — a genuine, meaningful gap, in contrast to Experiment 1
  where the two schemes were indistinguishable. This suggests the relative
  ranking of positional encoding schemes for length generalization is
  itself task-dependent, not fixed.
- **All three schemes still ultimately collapsed at sufficiently long
  lengths** (15+), consistent with Experiment 1 and with published
  findings — length generalization remains a real, unresolved limitation
  across every configuration tested here.
- **Qualitative failure mode (from an earlier single-length version of
  this experiment):** at long, unseen lengths, predictions clustered
  toward the small count values most common in the training distribution
  (mostly 0-1), rather than resembling random guessing — suggesting the
  model learned a distributional shortcut tied to training-time count
  ranges rather than a genuine length-invariant counting mechanism.

---

## Cross-experiment synthesis

1. **Whether positional encoding is necessary at all depends on the task's
   structure.** NoPE failed completely on the inherently positional
   reversal task, but partially succeeded on the less-positional counting
   task.
2. **Which specific encoding scheme performs best is also task-dependent.**
   Sinusoidal and learned PE tied on reversal, but learned PE meaningfully
   outperformed sinusoidal on counting.
3. **No configuration tested achieved robust length generalization.** Every
   scheme, on every task, degraded substantially by 15-25 tokens beyond
   the trained length or range — consistent with the broader literature's
   finding that length generalization remains an open problem in
   Transformer architectures.
4. **Experimental design matters**: training on a single fixed length
   (Experiment 1) versus a range of lengths (Experiment 2, after revision)
   changes what "length generalization failure" actually measures. A
   fixed-length-range setup, consistently applied, is recommended for any
   follow-up work extending this comparison.
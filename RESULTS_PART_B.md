# Part B — Length generalization experiment

**Research question:** does positional encoding scheme affect a from-scratch
transformer's ability to generalize to sequences longer than it was trained
on?

**Shared setup across all runs:** d_model=32, num_heads=4, d_ff=128,
num_layers=2, Adam optimizer (lr=0.001), trained for 1000 steps on
seq_len=6 only, evaluated (never trained) on lengths up to 25.

---

## Experiment 1: Reversal task (inherently positional)

**Task:** given a sequence of digits (0-9), output the sequence reversed.
Exact-match accuracy per position.

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
  accuracy even at the trained length (6), far below the ~100% both other
  schemes reached. Reversal is inherently positional — the correct output
  for a token depends entirely on its position, not just its identity —
  so a model with no way to represent position cannot solve it, regardless
  of sequence length. Not evaluated for length generalization since it
  never solved the underlying task.
- **Sinusoidal and learned positional encoding performed almost
  identically**: both reached perfect (1.000) accuracy at the trained
  length, and both collapsed to near-random performance (roughly 0.11-0.25,
  close to the ~0.10 random-guessing baseline for 10 classes) immediately
  past the training length, with no clear advantage for either scheme in
  the sharpness of the collapse or the post-collapse plateau.
- **Interpretation:** for this task, the meaningful dividing line was
  "having an explicit positional encoding at all" vs. "having none" (which
  determined whether the task could be learned in the first place) — not
  "which encoding scheme" (which made no observable difference to length
  generalization once the task was learned). Both working schemes failed
  to generalize past the training length in essentially the same way,
  suggesting the length generalization failure here is not specific to
  the encoding formula, but to how position is represented as an absolute,
  in-distribution-only signal in both cases.

---

## Experiment 2: [task TBD — less-positional task, e.g. counting digit occurrences]

*Planned: compare the same three encoding schemes on a task that does not
inherently require knowing token position, to test whether NoPE's failure
in Experiment 1 was specific to the positional nature of the reversal task,
and whether length generalization results differ once all three schemes
can actually learn the base task.*
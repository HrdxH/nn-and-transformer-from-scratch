# Neural Network & Transformer From Scratch

A two-part project building a feedforward neural network and a Transformer
encoder from first principles, culminating in an empirical study of length
generalization in Transformers. Full write-up: see `paper_draft.pdf`
("Task-Dependent Length Generalization in Transformers: An Empirical
Study").

## Part A: Neural Network from Scratch

- Fully connected network (784 -> 32 -> 32 -> 10), trained on MNIST
- Manual forward pass, backpropagation, and gradient descent in NumPy
  (no autograd)
- Diagnosed and fixed a "dying ReLU" activation collapse caused by naive
  weight initialization combined with an aggressive learning rate
- Ablation study: full-batch vs. mini-batch gradient descent vs. momentum
- Final result: 95.6% test accuracy

See `DESIGN_NOTES.md` and `RESULTS.md` (Part A) for full details, decisions,
and the complete experiment log.

**Notebook:** `part1_nn_from_scratch.ipynb`

## Part B: Length Generalization in a From-Scratch Transformer

A Transformer encoder (multi-head self-attention, feedforward sublayers,
residual connections, layer normalization) implemented from scratch in
PyTorch, used to investigate an open question in the literature: does a
model's positional encoding scheme affect its ability to generalize to
sequence lengths beyond those seen during training?

Three positional encoding schemes (sinusoidal, learned, and none) were
compared across two synthetic tasks chosen to differ in how strongly they
depend on token position:

- **Experiment 1 — Reversal** (inherently positional): reverse a sequence
  of digits.
- **Experiment 2 — Counting** (weakly positional): count occurrences of a
  target digit in a sequence.

**Headline finding:** both *whether* positional encoding is necessary and
*which* scheme performs best are task-dependent. No positional encoding
fails outright on reversal but partially succeeds on counting; sinusoidal
and learned encodings are indistinguishable on reversal but diverge
substantially on counting. No configuration achieves robust generalization
to substantially longer sequences.

See `DESIGN_NOTES_PART_B.md` and `RESULTS_PART_B.md` for full
methodology, including a mid-project revision from single-length to
range-based training, and complete results tables.

**Notebooks:**
- `part2_transformer_experiment1_reversal.ipynb`
- `part2_transformer_experiment2_counting.ipynb`
- `model_utils.py` — shared Transformer architecture components imported
  by both experiment notebooks.

## Paper

`paper_draft.pdf` / `paper_draft.docx` — full write-up covering both parts,
including related work (Vaswani et al., Press et al.'s ALiBi, Kazemnejad
et al.'s NoPE study, Ruoss et al.), a cross-experiment synthesis, and
stated limitations.

## Repository structure

```
.
├── part1_nn_from_scratch.ipynb
├── part2_transformer_experiment1_reversal.ipynb
├── part2_transformer_experiment2_counting.ipynb
├── model_utils.py
├── DESIGN_NOTES.md              # Part A
├── DESIGN_NOTES_PART_B.md       # Part B
├── RESULTS.md                   # Part A experiment log
├── RESULTS_PART_B.md            # Part B experiment log
└── README.md
```

## Setup

```bash
python3.14 -m venv venv
source venv/bin/activate
pip install numpy matplotlib scikit-learn jupyter pandas torch
```
# Experiment log

| Experiment | Config | Train acc | Test acc | Notes |
|---|---|---|---|---|
| Baseline | 16-16-10, sigmoid, lr=0.1 | 0.11 | 0.11 | Collapsed — dead ReLU |
| Fix 1 | 16-16-10, ReLU+He init, lr=0.01 | 0.81 | 0.78 | Stable, no collapse |
| Fix 2 | 32-32-10, ReLU+He init, lr=0.01 | 0.90 | 0.90 | Wider layers helped |
| Mini-batch | 32-32-10, batch=128, lr=0.01, 50 epochs | 0.9637 | 0.9552 | Huge improvement — ~15x more weight updates than full-batch despite far fewer epochs |
| Momentum | 32-32-10, batch=128, lr=0.01, beta=0.9, 50 epochs | 0.9634 | 0.9559 | No meaningful gain over mini-batch alone |
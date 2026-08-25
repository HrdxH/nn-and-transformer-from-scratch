# Design Notes — Part A: Neural Network from Scratch

- **Architecture**: 784 → 32 → 32 → 10, fully connected, trained on MNIST.
- **ReLU over sigmoid** for hidden layers, to avoid vanishing gradients in
  deeper networks (Glorot & Bengio, 2010).
- **Softmax + cross-entropy** for the output layer, standard for multi-class
  classification — gives a clean combined gradient (A3 - y_onehot).
- **He initialization** (scaling by sqrt(2/n_in)) instead of naive small
  random weights — needed to avoid dead ReLU neurons early in training
  (He et al., 2015). Naive initialization + high learning rate caused the
  network to collapse to predicting a single class.
- **Learning rate**: 0.1 was too aggressive (caused collapse); 0.01 was
  stable.
- **Result**: 90.1% train accuracy, 90.4% test accuracy after 1500 epochs,
  32-neuron hidden layers. No overfitting observed (test ≈ train).
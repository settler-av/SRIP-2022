# SRIP-2022

Implement two hidden layers neural network classifier from scratch in JAX [20 Marks]
---

1. Two hidden layers here means (input - hidden1 - hidden2 - output).
2. You must not use flax, optax, or any other library for this task.
3. Use MNIST dataset with 80:20 train:test split.
4. Manually optimize the number of neurons in hidden layers.
5. Use gradient descent from scratch to optimize your network. You should use the Pytree concept of JAX to do this elegantly.
6. plot loss v/s iterations curve with matplotlib.
7. evaluate the model on test data with various classification metrics and briefly discuss their implications.

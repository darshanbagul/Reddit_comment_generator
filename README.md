# Generating text using a Gated Recurrent Unit

In this project, I have implemented a Gated Recurrent Unit (GRU) RNN using theano for generating reddit comments. This project helped me understand the intricate differences between LSTMs and GRUs, while experimenting with Theano.

## Gated Recurrent Units (GRUs)

Gated Recurrent Units are a variant of Recurrent Neural Network which were developed with an idea quite similar to LSTM Networks - solve the vanishing gradient problem and help RNNs learn representations across multiple time slices. A GRU has two gates, a reset gate r, and an update gate z.  Intuitively, the reset gate determines how to combine the new input with the previous memory, and the update gate defines how much of the previous memory to keep around. If we set the reset to all 1’s and  update gate to all 0’s we again arrive at our plain RNN model. Below are the equations: 

          z(t) = σ(x(t) * Uz + s(t-1) * Wz)
          r(t) = σ(x(t) * Ur + s(t-1) * Wr)
          h(t) = tanh(x(t) * Uh + (s(t-1) .* r) * Wh)
          s(t) = (1-z) .* h + z .* s(t-1)
          
The basic idea of using a gating mechanism to learn long-term dependencies by countering the vanishing/exploding gradients.

## GRU vs LSTMs

How do GRUs differ from LSTMs?
  1. A GRU cell has 2 gates, whereas an LSTM cell has 3 gates.
  2. GRUs don't have an internal memory state different from exposed hidden state. They don’t have the output gate that is present in LSTMs.
  3. The input and forget gates are coupled by an update gate z and the reset gate r is applied directly to the previous hidden state, splitting the reset task of LSTM between these two gates.
  4. In LSTMs we apply another non-linear function to calculate outputs; which we don't for GRUs.
  
GRUs have fewer parameters as U and W weight matrices are smaller; which can contribute to bit faster training times as well as requiring less data for generalization. On the other hand, if you have enough data, the greater expressive power of LSTMs may lead to better results.

In many tasks both architectures yield comparable performance and tuning hyperparameters like layer size is probably more important than picking the ideal architecture. Hence there is no clear winner, as stated by the empirical evaluations in [Empirical Evaluation of Gated Recurrent Neural Networks on Sequence Modeling](https://arxiv.org/abs/1412.3555)  and [An Empirical Exploration of Recurrent Network Architectures](http://proceedings.mlr.press/v37/jozefowicz15.pdf)

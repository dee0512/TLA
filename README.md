# Temporally Layered Architecture
Author's PyTorch implementation of Temporally Layered Architecture for OpenAI gym tasks.

Following are the code dependencies:
1. [`Python 3.7`](https://www.python.org/)
2. [`PyTorch                  1.12.0`](https://pytorch.org/)
3. [`gym 0.21.0`](https://github.com/openai/gym)
4. [`mujoco-py              2.0.2.13`](http://roboti.us/download.html)

### Usage

In order to reproduce paper, follow the following three part process:

1. run `train_slow_layer.py` to train the slow layer. 
2. run `train_fast_layer.py` to train the fast layer. Remember to set the `parent_response_rate` equal to the one set in the slow layer.
3. To evaluate the code, run `evaluate.py`.

The default hyperparameters are set for the `InvertedPendulum-v2` envrionment. 

# inverse_nn
Experiments exploring the reversibility of neural networks

## Example execution

python random_mlp_recovery.py --input 1000 --hidden 1000 --layers 1 --iterations 100000 --lr .1 --activation relu --optimizer AdamOptimizer

Choices for activation funciton include:
"relu", "elu", and "tanh".

Choices for optimizer include
"AdamOptimizer" and "GradientDescentOptimizer"

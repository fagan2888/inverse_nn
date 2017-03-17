
import numpy as np
import tensorflow as tf
import argparse


parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--input', dest="INPUT_DIMENSION", type=int, default=100, help='input dimension')
parser.add_argument('--hidden', dest="HIDDEN_DIMENSION", type=int, default=100, help='hidden dimension')
parser.add_argument('--layers', dest="HIDDEN_LAYERS", type=int, default=1, help='hidden layers')
parser.add_argument('--iterations', dest="ITERATIONS", type=int, default=10000, help='number of iterations')
parser.add_argument('--activation', dest="ACTIVATION", type=str, default="relu", help='activation function')
parser.add_argument('--optimizer', dest="OPTIMIZER", type=str, default="GradientDescentOptimizer", help='choice of optimizer')
parser.add_argument('--lr', dest="LEARNING_RATE", type=float, default=.01, help='learning rate')

args = parser.parse_args()


act_dict={
    "relu": tf.nn.relu,
    "elu": tf.nn.elu,
    "tanh": tf.nn.tanh,
}

opt_dict={
    "AdamOptimizer": tf.train.AdamOptimizer,
    "GradientDescentOptimizer": tf.train.GradientDescentOptimizer
}

print("INPUT_DIMENSION: %s" % (args.INPUT_DIMENSION))

LEARNING_RATE = args.LEARNING_RATE
INPUT_DIMENSION = args.INPUT_DIMENSION
HIDDEN_DIMENSION = args.HIDDEN_DIMENSION
NUM_LAYERS = args.HIDDEN_LAYERS
ITERATIONS = args.ITERATIONS
act = act_dict[args.ACTIVATION]
opt = opt_dict[args.OPTIMIZER]

z       = tf.constant(np.random.random(INPUT_DIMENSION).astype("float32").reshape((1,INPUT_DIMENSION)) * 2 - 1)
z_prime = tf.Variable(tf.random_uniform([1, INPUT_DIMENSION], minval=-1, maxval=1))

weights = []

W1 = tf.constant(np.random.randn(INPUT_DIMENSION* HIDDEN_DIMENSION).astype("float32").reshape((INPUT_DIMENSION, HIDDEN_DIMENSION)) * .01)
b1 = tf.constant(np.random.randn(HIDDEN_DIMENSION).astype("float32") *.01)

weights.append((W1, b1))

for i in range(NUM_LAYERS-1):
    Wi = tf.constant(np.random.randn(HIDDEN_DIMENSION * HIDDEN_DIMENSION).astype("float32").reshape((HIDDEN_DIMENSION, HIDDEN_DIMENSION)) *.01)
    bi = tf.constant(np.random.randn(HIDDEN_DIMENSION).astype("float32") *.01)

    weights.append((Wi, bi))

def G(x):
    h = x
    for i in range(NUM_LAYERS):
        W, b = weights[i]
        h = act(tf.matmul(h, W)+b)
    return h


Gz = G(z)
Gz_prime = G(z_prime)

cost = tf.reduce_mean((Gz - Gz_prime) * (Gz -Gz_prime))
z_cost = tf.reduce_mean((z - z_prime) * (z - z_prime))

L1G = tf.reduce_mean(tf.abs(Gz - Gz_prime))
L1z = tf.reduce_mean(tf.abs(z - z_prime))


optimizer = opt(learning_rate=LEARNING_RATE).minimize(cost, var_list=[z_prime])

gradients = tf.gradients(cost, [z_prime])
grad_norm = tf.reduce_mean(gradients[0] * gradients[0])


init = tf.global_variables_initializer()

sess = tf.Session()
sess.run(init)


if __name__ == "__main__":
    gradient_list =[]
    zp_list = []
    for i in range(ITERATIONS):
        _, G_error, z_error, real_L1G, real_L1z, norm, grads, zp = sess.run([optimizer, cost, z_cost, L1G, L1z, grad_norm, gradients, z_prime])
        gradient_list.append(grads[0])
        zp_list.append(zp)
        print("iteration: %s, G-loss: %s, Z-loss: %s, L1G: %s, L1z: %s, L1G/L1z: %s, grad_norm: %s" % (i, G_error, z_error, real_L1G, real_L1z, real_L1G/real_L1z, norm))



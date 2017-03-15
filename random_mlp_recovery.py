
import numpy as np
import tensorflow as tf

learning_rate = 10000

INPUT_DIMENSION = 100
HIDDEN_DIMENSION = 100000
NUM_LAYERS = 1
ITERATIONS = 10000

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
        h= tf.nn.relu(tf.matmul(h, W)+b)
    return h


Gz = G(z)
Gz_prime = G(z_prime)

cost = tf.reduce_mean((Gz - Gz_prime) * (Gz -Gz_prime))
z_cost = tf.reduce_mean((z - z_prime) * (z - z_prime))

L1G = tf.reduce_mean(tf.abs(Gz - Gz_prime))
L1z = tf.reduce_mean(tf.abs(z - z_prime))


optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)


init = tf.global_variables_initializer()

sess = tf.Session()
sess.run(init)

def run(iterations):
    for i in range(iterations):
        _, G_error, z_error, real_L1G, real_L1z = sess.run([optimizer, cost, z_cost, L1G, L1z])
        print("iteration: %s, G-loss: %s, Z-loss: %s, L1G: %s, L1z: %s, L1G/L1z: %s" % (i, G_error, z_error, real_L1G, real_L1z, real_L1G/real_L1z))


if __name__ == "__main__":
    run(10000)



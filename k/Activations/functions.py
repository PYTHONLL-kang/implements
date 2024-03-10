import tensorflow as tf

class Sigmoid:
    def forward(self, data):
        return tf.math.divide(1, (1 + tf.math.exp(-data)))

    def backward(self, gradients, inputs, outputs):
        activation_gradients = tf.math.multiply(outputs, tf.math.subtract(1, outputs))
        return tf.math.multiply(gradients, activation_gradients)

class ReLU:
    def forward(self, data):
        return tf.math.maximum(0, data)

    def backward(self, gradients, inputs, outputs):
        activation_gradients = tf.math.divide(outputs, inputs+1e-7)
        return tf.math.multiply(gradients, activation_gradients)

class Linear:
    def forward(self, data):
        return data

    def backward(self, gradients, inputs, outputs):
        return gradients

class Softmax:
    def forward(self, data):
        e_x = tf.math.exp(data - tf.math.reduce_max(data))
        return e_x / tf.math.reduce_sum(e_x, axis=1, keepdims=True)

    def backward(self, gradients, inputs, outputs):
        reshaped_outputs = tf.expand_dims(outputs, axis=-1)
        outer_outputs = tf.matmul(reshaped_outputs, tf.transpose(reshaped_outputs, [0, 2, 1]))
        activation_gradients = tf.linalg.diag(outputs) - outer_outputs
        return tf.reduce_sum(activation_gradients * tf.expand_dims(gradients, axis=-1), axis=1)

class Tanh:
    def forward(self, data):
        return tf.math.exp(data) - tf.math.exp(-data) / tf.math.exp(data) + tf.math.exp(-data)

    def backward(self, gradients, inputs, outputs):
        activation_gradients = 1 - tf.math.pow(outputs, 2)
        return tf.math.multiply(activation_gradients, gradients)

def function(name):
    function_map = {'sigmoid' : Sigmoid, 'relu' : ReLU,
                    'linear' : Linear, 'softmax' : Softmax,
                    'tanh' : Tanh, 'hyperbolic_tangent' : Tanh}

    return function_map[name.lower()]()
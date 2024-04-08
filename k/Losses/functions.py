import tensorflow as tf

EPSILON = 1e-10

class MeanSquaredError:
    def forward(self, true, pred):
        return tf.math.reduce_mean(tf.math.pow(pred - true, 2))

    def backward(self, true, pred):
        return 2*(pred - true)

class BinaryCrossEntropy:
    def forward(self, true, pred):
        pred = tf.clip_by_value(pred, EPSILON, 1-EPSILON)
        return true * tf.math.log(pred) + (1-true) * tf.math.log(1-pred)

    def backward(self, true, pred):
        pred += EPSILON
        return -tf.math.subtract(true / pred, (1-true) / (1-pred))

class CategoricalCrossEntropy:
    def forward(self, true, pred):
        pred = tf.clip_by_value(pred, EPSILON, 1-EPSILON)
        return -tf.math.reduce_mean(true * tf.math.log(pred))

    def backward(self, true, pred):
        return -tf.math.divide(true, pred+EPSILON)

class MeanAbsoluteError:
    def forward(self, true, pred):
        error = pred - true
        return tf.math.reduce_mean(error * tf.math.sign(error))

    def backward(self, true, pred):
        error = pred - true
        return tf.math.sign(error)

def function(name):
    function_map = {'mse' : MeanSquaredError, 'meansquarederror' : MeanSquaredError,
                    'bce' : BinaryCrossEntropy, 'binarycrossentropy' : BinaryCrossEntropy,
                    'cce' : CategoricalCrossEntropy, 'categoricalcrossentropy' : CategoricalCrossEntropy,
                    'mae' : MeanAbsoluteError, 'meanabsoluteerror' : MeanAbsoluteError}

    return function_map[name.lower()]()
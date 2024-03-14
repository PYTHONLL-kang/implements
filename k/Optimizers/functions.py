import tensorflow as tf

class SGD:
    def __init__(self, **kwargs):
        self.learning_rate = kwargs.get('learning_rate', 1e-4)

    def update(self, params, gradients):
        params.assign_sub(self.learning_rate * gradients)

        return params

class Momentum:
    def __init__(self, **kwargs):
        self.learning_rate = kwargs.get('learning_rate', 1e-4)
        self.decay_rate = kwargs.get('decay_rate', 0.9)
        self.accumulated = 0

    def update(self, params, gradients):
        self.accumulated = self.decay_rate * self.accumulated + self.learning_rate * gradients
        params.assign_sub(self.accumulated)

        return params

class Adagrad:
    def __init__(self, **kwargs):
        self.learning_rate = kwargs.get('learning_rate', 1e-3)
        self.epsilon = kwargs.get('epsilon', 1e-8)
        self.accumulated = 0

    def update(self, params, gradients):
        self.accumulated += tf.math.pow(gradients, 2)
        params.assign_sub(self.learning_rate * gradients / (self.epsilon + tf.math.sqrt(self.accumulated)))

        return params

class RMSprop:
    def __init__(self, **kwargs):
        self.learning_rate = kwargs.get('learning_rate', 1e-4)
        self.decay_rate = kwargs.get('decay_rate', 0.9)
        self.epsilon = kwargs.get('epsilon', 1e-8)
        self.accumulated = 0

    def update(self, params, gradients):
        self.accumulated = self.decay_rate * self.accumulated + (1 - self.decay_rate) * tf.math.pow(gradients, 2)
        params.assign_sub(self.learning_rate * gradients / (self.epsilon + tf.math.sqrt(self.accumulated)))

        return params

class Adam:
    def __init__(self, **kwargs):
        self.learning_rate = kwargs.get('learning_rate', 1e-3)
        self.beta1 = kwargs.get('beta1', 0.9)
        self.beta2 = kwargs.get('beta2', 0.999)
        self.epsilon = kwargs.get('epsilon', 1e-8)

        self.m = 0
        self.v = 0
        self.t = 0

    def update(self, params, gradients):
        self.t += 1

        self.m = self.beta1 * self.m + (1 - self.beta1) * gradients
        self.v = self.beta2 * self.v + (1 - self.beta2) * tf.math.pow(gradients, 2)
        m_hat = self.m / (1 - self.beta1 ** self.t)
        v_hat = self.v / (1 - self.beta2 ** self.t)
        params.assign_sub(self.learning_rate * m_hat / (self.epsilon + tf.math.sqrt(v_hat)))
            
        return params

def function(name):
    function_map = {'sgd' : SGD, 'momentum' : Momentum,
                    'adagrad' : Adagrad, 'RMSprop' : RMSprop,
                    'adam' : Adam}

    return function_map[name.lower()]()
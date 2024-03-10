import tensorflow as tf

class SGD:
    def __init__(self, **kwargs):
        self.learning_rate = kwargs.get('learning_rate', 1e-4)

    def update(self, params, gradients):
        for i in range(len(params)):
            params[i].assign_sub(self.learning_rate * gradients[i])

        return params

class Momentum:
    def __init__(self, **kwargs):
        self.learning_rate = kwargs.get('learning_rate', 1e-4)
        self.decay_rate = kwargs.get('decay_rate', 0.9)
        self.accumulated = None

    def update(self, params, gradients):
        if self.accumulated is None:
            self.accumulated = [0] * len(params)

        for i in range(len(params)):
            self.accumulated[i] = self.decay_rate * self.accumulated[i] + self.learning_rate * gradients[i]
            params[i].assign_sub(self.accumulated[i])

        return params

class Adagrad:
    def __init__(self, **kwargs):
        self.learning_rate = kwargs.get('learning_rate', 1e-3)
        self.epsilon = kwargs.get('epsilon', 1e-8)
        self.accumulated = None

    def update(self, params, gradients):
        if self.accumulated is None:
            self.accumulated = [0] * len(params)
        
        for i in range(len(params)):
            self.accumulated[i] += tf.math.pow(gradients[i], 2)
            params[i].assign_sub(self.learning_rate * gradients[i] / (self.epsilon + tf.math.sqrt(self.accumulated[i])))

        return params

class RMSprop:
    def __init__(self, **kwargs):
        self.learning_rate = kwargs.get('learning_rate', 1e-4)
        self.decay_rate = kwargs.get('decay_rate', 0.9)
        self.epsilon = kwargs.get('epsilon', 1e-8)
        self.accumulated = None

    def update(self, params, gradients):
        if self.accumulated is None:
            self.accumulated = [0] * len(params)

        for i in range(len(params)):
            self.accumulated[i] = self.decay_rate * self.accumulated[i] + (1 - self.decay_rate) * tf.math.pow(gradients[i], 2)
            params[i].assign_sub(self.learning_rate * gradients[i] / (self.epsilon + tf.math.sqrt(self.accumulated[i])))

        return params

class Adam:
    def __init__(self, **kwargs):
        self.learning_rate = kwargs.get('learning_rate', 1e-3)
        self.beta1 = kwargs.get('beta1', 0.9)
        self.beta2 = kwargs.get('beta2', 0.999)
        self.epsilon = kwargs.get('epsilon', 1e-8)

        self.m = None
        self.v = None
        self.t = 0

    def update(self, params, gradients):
        if self.m is None:
            self.m = [0] * len(params)
            self.v = [0] * len(params)
        
        self.t += 1
        for i in range(len(params)):
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * gradients[i]
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * tf.math.pow(gradients[i], 2)
            m_hat = self.m[i] / (1 - self.beta1 ** self.t)
            v_hat = self.v[i] / (1 - self.beta2 ** self.t)
            params[i].assign_sub(self.learning_rate * m_hat / (self.epsilon + tf.math.sqrt(v_hat)))
            
        return params

def function(name):
    function_map = {'sgd' : SGD, 'momentum' : Momentum,
                    'adagrad' : Adagrad, 'RMSprop' : RMSprop,
                    'adam' : Adam}

    return function_map[name.lower()]()
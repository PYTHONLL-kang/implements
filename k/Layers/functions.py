import tensorflow as tf
import k.Layers.base_functions as Layers

class Dense(Layers.Base):
    def __init__(self, output_shape, **kwargs):
        self.input_shape = kwargs.get('input_shape')
        self.output_shape = output_shape
        super().__init__(**kwargs)

    def set_shape(self):
        self.weight_shape = [self.input_shape, self.output_shape]
    
    def forward(self, data):
        matmul_output = tf.matmul(data, self.weight)
        return tf.math.add(matmul_output, self.bias)

    def backward(self, gradients, inputs, outputs):
        gradients = self.process_gradients(gradients)

        weight_gradients = tf.matmul(tf.transpose(inputs), gradients)
        bias_gradients = tf.math.reduce_sum(gradients, axis=0, keepdims=True)

        param = self.optimizer.update([self.weight, self.bias], [weight_gradients, bias_gradients])
        self.weight.assign(param[0])
        self.bias.assign(param[1])

        return tf.matmul(gradients, tf.transpose(self.weight))

class Convolution2D(Layers.Base, Layers.Image):
    def __init__(self, filters, kernel_shape, **kwargs):
        self.input_shape = kwargs.get('input_shape')
        kwargs['Initializer'] = kwargs.get('Initializer', 'xavier')

        self.kernel_shape = kernel_shape
        self.filters = filters

        Layers.Base.__init__(self, **kwargs)
        Layers.Image.__init__(self, **kwargs)

    def set_shape(self):
        h, w, _, __ = self.set_imgshape(self.input_shape[0], self.input_shape[1])
        self.output_shape = [h, w, self.filters]
        self.weight_shape = [self.kernel_shape[0], self.kernel_shape[1], self.input_shape[-1], self.filters]

    def forward(self, data):
        self.col, num, out_h, out_w = self.im2col(data)
        self.col_weight = tf.reshape(self.weight, (tf.reduce_prod(self.weight_shape) // self.filters, -1))

        col_output = tf.matmul(self.col, self.col_weight)
        convolution_output = tf.reshape(col_output, (num, out_h, out_w, self.filters))

        return tf.math.add(convolution_output, self.bias)

    def backward(self, gradients, inputs, outputs):
        gradients = self.process_gradients(gradients)
        col_gradients = tf.reshape(gradients, (-1, self.filters))

        weight_gradients = tf.reshape(tf.matmul(tf.transpose(self.col), col_gradients), self.weight_shape)
        bias_gradients = tf.math.reduce_sum(col_gradients, axis=0, keepdims=True)

        param = self.optimizer.update([self.weight, self.bias], [weight_gradients, bias_gradients])
        self.weight.assign(param[0])
        self.bias.assign(param[1])

        col_weight_gradients = tf.matmul(col_gradients, tf.transpose(self.col_weight))

        return self.col2im(col_weight_gradients, inputs)

class Pooling2D(Layers.Image):
    def __init__(self, pool_type, pool=(2, 2), **kwargs):
        self.input_shape = None

        self.pool_type = pool_type
        self.pool_size = pool[0] * pool[1]
        self.kernel_shape = pool

        super().__init__(**kwargs)

    def set_shape(self):
        h, w, _, __ = self.set_imgshape(self.input_shape[1], self.input_shape[2])
        self.output_shape = [h, w, input_shape[-1]]

    def forward(self, data):
        self.col, num, out_h, out_w = self.im2col(data)
        
        if self.pool_type == 'max':
            pooling_col = tf.math.reduce_max(self.col, axis=0, keepdims=True)

        if self.pool_type == 'mean':
            pooling_col = tf.math.reduce_mean(self.col, axis=0, keepdims=True)

        pooling_data = tf.reshape(pooling_col, (num, self.kernel_shape[0], self.kernel_shape[1], data.shape[-1]))

        return pooling_data

    def backward(self, gradients, inputs, outputs):
        layer_gradients = tf.math.multiply(outputs, gradients)

        col_layer_gradients = tf.zeros((tf.size(layer_gradients), self.pool_size))

        if self.pool_type == 'max':
            index = tf.reshape(tf.argmax(self.col, axis=0), (-1))
            col_layer_gradients = tf.tensor_scatter_nd_update(col_layer_gradients, tf.expand_dims(tf.range(self.pool_size), axis=1), tf.reshape(layer_gradients, (-1)))

        if self.pool_type == 'mean':
            col_layer_gradients.fill(tf.reduce_mean(layer_gradients))
            col_layer_gradients = tf.broadcast_to(col_layer_gradients, layer_gradients.shape + (self.pool_size,))

        col_layer_gradients = tf.reshape(col_layer_gradients, (layer_gradients.shape + (self.pool_size,)))
        col_gradients = tf.reshape(col_layer_gradients, (layer_gradients.shape[0]*layer_gradients.shape[1]*layer_gradients.shape[2], -1))

        return self.col2im(col_gradients, inputs)

class Flatten:
    def __init__(self):
        self.input_shape = None

    def set_shape(self):
        self.output_shape = tf.reduce_prod(self.input_shape)

    def forward(self, data):
        return tf.reshape(data, (data.shape[0], -1))

    def backward(self, gradients, inputs, outputs):
        return tf.reshape(tf.math.multiply(outputs, gradients), inputs.shape)
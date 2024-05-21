import tensorflow as tf
import k.Layers.base_functions as Layers
import k.Activations.functions as Activations

class Input():
    def __init__(self, input_shape):
        self.output_shape = input_shape

class Dense(Layers.Base):
    def __init__(self, output_shape, **kwargs):
        self.input_shape = kwargs.get('input_shape')
        self.output_shape = output_shape
        super().__init__(**kwargs)

    def set_shape(self):
        self.weight_shape = [self.input_shape, self.output_shape]
    
    def forward(self, data):
        output = tf.matmul(data, self.weight)

        if self.use_bias:
            output = tf.math.add(output, self.bias)

        return output

    def backward(self, gradients, inputs, outputs):
        weight_gradients = tf.matmul(tf.transpose(inputs), gradients)
        self.weight.assign(self.weight_optimizer.update(self.weight, weight_gradients))

        if self.use_bias:
            bias_gradients = tf.math.reduce_sum(gradients, axis=0, keepdims=True)
            self.bias.assign(self.bias_optimizer.update(self.bias, bias_gradients))

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
        output = tf.reshape(col_output, (num, out_h, out_w, self.filters))

        if self.use_bias:
            output = tf.math.add(output, self.bias)

        return output

    def backward(self, gradients, inputs, outputs):
        col_gradients = tf.reshape(gradients, (-1, self.filters))

        weight_gradients = tf.reshape(tf.matmul(tf.transpose(self.col), col_gradients), self.weight_shape)
        self.weight.assign(self.weight_optimizer.update(self.weight, weight_gradients))

        if self.use_bias:
            bias_gradients = tf.math.reduce_sum(col_gradients, axis=0, keepdims=True)
            self.bias.assign(self.bias_optimizer.update(self.bias, bias_gradients))

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
        h, w, _, __ = self.set_imgshape(self.input_shape[0], self.input_shape[1])
        self.output_shape = [h, w, self.input_shape[-1]]

    def forward(self, data):
        self.col, num, out_h, out_w = self.im2col(data)
        self.col = tf.reshape(self.col, (num*out_h*out_w*data.shape[-1], -1))

        if self.pool_type == 'max':
            pooling_col = tf.math.reduce_max(self.col, axis=1, keepdims=True)

        if self.pool_type == 'mean':
            pooling_col = tf.math.reduce_mean(self.col, axis=1, keepdims=True)

        pooling_data = tf.reshape(pooling_col, (num, out_h, out_w, data.shape[-1]))

        return pooling_data

    def backward(self, gradients, inputs, outputs):
        layer_gradients = tf.math.multiply(outputs, gradients)

        col_layer_gradients = tf.zeros((tf.size(layer_gradients), self.pool_size))

        if self.pool_type == 'max':
            argmax_index = tf.reshape(tf.argmax(self.col, axis=1), (-1))
            index = tf.Variable([tf.range(argmax_index.shape[0], dtype=tf.int64), argmax_index])
            col_layer_gradients = tf.tensor_scatter_nd_update(col_layer_gradients, tf.transpose(index), tf.reshape(layer_gradients, (-1)))

        if self.pool_type == 'mean':
            col_layer_gradients = tf.fill(col_layer_gradients.shape, tf.reduce_mean(layer_gradients))
            col_layer_gradients = tf.reshape(col_layer_gradients, (layer_gradients.shape + (self.pool_size,)))

        col_layer_gradients = tf.reshape(col_layer_gradients, (layer_gradients.shape + (self.pool_size,)))
        col_gradients = tf.reshape(col_layer_gradients, (layer_gradients.shape[0]*layer_gradients.shape[1]*layer_gradients.shape[2], -1))

        return tf.cast(self.col2im(col_gradients, inputs), dtype=tf.float32)

class Flatten:
    def __init__(self):
        self.input_shape = None

    def set_shape(self):
        self.output_shape = tf.reduce_prod(self.input_shape)

    def forward(self, data):
        return tf.reshape(data, (data.shape[0], -1))

    def backward(self, gradients, inputs, outputs):
        return tf.reshape(tf.math.multiply(outputs, gradients), inputs.shape)

class Recurrent_cell:
    def __init__(self, input_shape, output_shape, **kwargs):
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.hidden_shape = self.output_shape

        self.ih_layer = Dense(input_shape=self.input_shape, output_shape=self.hidden_shape, use_bias=False, **kwargs)
        self.hh_layer = Dense(input_shape=self.hidden_shape, output_shape=self.hidden_shape, use_bias=True, **kwargs)
        self.ho_layer = Dense(input_shape=self.hidden_shape, output_shape=self.output_shape, use_bias=True, **kwargs)

        self.activations = Activations.Tanh()

    def set_vars(self, optimizer):
        self.ih_layer.set_shape()
        self.hh_layer.set_shape()
        self.ho_layer.set_shape()

        self.ih_layer.set_vars(optimizer)
        self.hh_layer.set_vars(optimizer)
        self.ho_layer.set_vars(optimizer)

    def forward(self, inputs, hiddens):
        a = self.ih_layer.forward(inputs) + self.hh_layer.forward(hiddens)
        h = self.activations.forward(a)
        o = self.ho_layer.forward(h)

        return h, o

    def backward(self, gradients, inputs, pre_hiddens, cur_hiddens, outputs):
        ho_gradients = self.ho_layer.backward(gradients, cur_hiddens, outputs)
        h_gradients = self.activations.backward(ho_gradients, inputs, outputs)

        input_graidents = self.ih_layer.backward(h_gradients, inputs, pre_hiddens)
        hidden_graidents = self.hh_layer.backward(h_gradients, pre_hiddens, cur_hiddens)

        return input_graidents, hidden_graidents

class Recurrent_network:
    def __init__(self, output_shape, input_shape, **kwargs):
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.timestep = self.input_shape[0]

        self.return_sequences = kwargs.get('return_sequences', False)
        self.return_states = kwargs.get('return_states', False)

        self.network = []
        self.hidden_shape = output_shape
        self.network.append(Recurrent_cell(input_shape[1], self.hidden_shape))
        for i in range(self.timestep-2):
            self.network.append(Recurrent_cell(input_shape[1], self.hidden_shape))
        self.network.append(Recurrent_cell(input_shape[1], output_shape))

    def set_shape(self):
        if self.return_sequences:
            self.output_shape = (self.timestep, self.output_shape)
        else:
            self.output_shape = self.output_shape

    def set_vars(self, optimizer):
        for i in range(self.timestep):
            self.network[i].set_vars(optimizer)
    
    def forward(self, data):
        o = []
        h = [tf.zeros((data.shape[0], self.hidden_shape))]
        for i in range(self.timestep):
            oi, hi = self.network[i].forward(data[:,i,:], h[i])

            o.append(oi)
            h.append(hi)

        self.hiddens = h
        o = tf.Variable(o)

        if not self.return_sequences:
            o = o[-1]

        if self.return_states:
            return o, h

        return o

    def backward(self, gradients, inputs, outputs):
        for i in reversed(range(self.timestep)):
            if self.return_sequences:
                o = outputs[:,i,:]
            else:
                o = outputs

            input_gradients, gradients = self.network[i].backward(gradients, inputs[:,i,:], self.hiddens[i-1], self.hiddens[i], o)
        
        return input_gradients
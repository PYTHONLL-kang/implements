import tensorflow as tf
import copy
import k.Initializers.functions as Initializers

class Base:
    def __init__(self, **kwargs):
        self.initializer = kwargs.get('initializer', 'he')
        self.dropout_rate = kwargs.get('dropout', 0.0)
        self.use_bias = kwargs.get('use_bias', True)

    def set_vars(self, optimizer):
        weight_stddev = Initializers.function(self.initializer, self.weight_shape)
        self.weight = tf.Variable(tf.random.normal(shape=self.weight_shape, mean=0, stddev=weight_stddev, dtype=tf.float32))
        self.weight_optimizer = copy.deepcopy(optimizer)

        if self.use_bias:
            self.bias = tf.Variable(tf.random.normal(shape=(1, self.weight_shape[-1]), mean=0, stddev=0.01, dtype=tf.float32))
            self.bias_optimizer = copy.deepcopy(optimizer)

class Image:
    def __init__(self, **kwargs):
        self.pad = self.set_padding(kwargs.get('padding', 'valid'))
        self.pad_value = kwargs.get('padding_value', 0)
        self.strides = kwargs.get('strides', (1, 1))

    def set_padding(self, padding):
        if type(padding) == tuple or type(padding) == list:
            return padding

        elif type(padding) == str:
            if padding == 'same':
                padding_height = self.strides[0] * (self.input_shape[0] - 1) - self.input_shape[0] + self.kernel_shape[0]
                padding_width = self.strides[1] * (self.input_shape[1] - 1) - self.input_shape[1] + self.kernel_shape[1]

                return [[0, 0], [padding_height, padding_height], [padding_width, padding_width], [3, 0]]

            if padding == 'valid':
                return [[0, 0], [0, 0], [0, 0], [0, 0]]

        else:
            raise ValueError("padding is not valid parameter")

    def set_imgshape(self, height, width):
        out_h = (height + 2 * self.pad[2][1] - self.kernel_shape[0]) // self.strides[0] + 1
        out_w = (width + 2 * self.pad[3][1] - self.kernel_shape[1]) // self.strides[1] + 1
        img_h = height + 2 * self.pad[2][1] + self.strides[0] -1
        img_w = width + 2 * self.pad[3][1] + self.strides[1] -1

        return out_h, out_w, img_h, img_w

    def im2col(self, data):
        num, height, width, channel = data.shape
        out_h, out_w, _, _ = self.set_imgshape(height, width)
        data = tf.cast(tf.transpose(tf.pad(data, self.pad, mode='CONSTANT', constant_values=self.pad_value), (0, 3, 1, 2)), dtype=tf.float32)
        col = tf.Variable(tf.zeros((num, channel, self.kernel_shape[0], self.kernel_shape[1], out_h, out_w), dtype=tf.float32))

        for y in range(self.kernel_shape[0]):
            y_max = y + self.strides[0] * out_h
            for x in range(self.kernel_shape[1]):
                x_max = x + self.strides[1] * out_w
                col[:, :, y, x, :, :].assign(data[:, :, y:y_max:self.strides[0], x:x_max:self.strides[1]])

        col = tf.reshape(tf.transpose(col, (0, 4, 5, 1, 2, 3)), shape=(num * out_h * out_w, -1))
        return col, num, out_h, out_w

    def col2im(self, col, data):
        num, height, width, channel = data.shape
        out_h, out_w, img_h, img_w = self.set_imgshape(height, width)
        col = tf.transpose(tf.reshape(col, (num, out_h, out_w, channel, self.kernel_shape[0], self.kernel_shape[1])), (0, 3, 4, 5, 1, 2))

        img = tf.Variable(tf.zeros((num, channel, img_h, img_w)))

        for y in range(self.kernel_shape[0]):
            y_max = y + self.strides[0] * out_h
            for x in range(self.kernel_shape[1]):
                x_max = x + self.strides[1] * out_w
                img[:, :, y:y_max:self.strides[0], x:x_max:self.strides[1]].assign(col[:, :, y, x, :, :])

        return tf.transpose(img, (0, 2, 3, 1))

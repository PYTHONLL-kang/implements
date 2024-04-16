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
    
class SSIM:
    def gaussian(window_size, sigma=1.5):
        gauss = tf.Variable([tf.math.exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)], dtype=tf.float16)
        return gauss / tf.math.reduce_sum(gauss)

    def create_window(window_size, channel):
        _1D_window = tf.expand_dims(SSIM.gaussian(window_size), axis=1)
        _2D_window = tf.expand_dims(tf.expand_dims(tf.linalg.matmul(_1D_window, _1D_window, transpose_b=True), axis=0), axis=0)
        window = tf.Variable(tf.broadcast_to(_2D_window, shape=(channel, 1, window_size, window_size)), dtype=tf.float16)

        return tf.transpose(window, perm=(3,2,1,0))

    def SSIM(img1, img2, k1=0.01, k2=0.02, L=1):
        img1 = tf.cast(img1, tf.float16)
        img2 = tf.cast(img2, tf.float16)

        window = SSIM.create_window(11, 3)

        mu1 = tf.nn.conv2d(img1, window, strides=[1, 1, 1, 1], padding="SAME")
        mu2 = tf.nn.conv2d(img2, window, strides=[1, 1, 1, 1], padding="SAME")

        mu1_sq = tf.square(mu1)
        mu2_sq = tf.square(mu2)
        mu1_mu2 = mu1 * mu2

        sigma1_sq = tf.nn.conv2d(tf.square(img1), window, strides=[1, 1, 1, 1], padding="SAME") - mu1_sq
        sigma2_sq = tf.nn.conv2d(tf.square(img2), window, strides=[1, 1, 1, 1], padding="SAME") - mu2_sq
        sigma12 = tf.nn.conv2d(img1 * img2, window, strides=[1, 1, 1, 1], padding="SAME") - mu1_mu2

        C1 = 0.01**2
        C2 = 0.03**2

        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

        return 1 - tf.reduce_mean(ssim_map)

def function(name):
    function_map = {'mse' : MeanSquaredError, 'meansquarederror' : MeanSquaredError,
                    'bce' : BinaryCrossEntropy, 'binarycrossentropy' : BinaryCrossEntropy,
                    'cce' : CategoricalCrossEntropy, 'categoricalcrossentropy' : CategoricalCrossEntropy,
                    'mae' : MeanAbsoluteError, 'meanabsoluteerror' : MeanAbsoluteError,
                    'ssim' : SSIM}

    return function_map[name.lower()]()
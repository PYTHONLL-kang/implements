import tensorflow as tf

def xavier(shape):
    return tf.math.sqrt(2 / sum(shape))

def he(shape):
    return tf.math.sqrt(1 / shape[0])

def lecun(shape):
    return tf.math.sqrt(2 / shape[0])

def function(name, shape):
    function_map = {'xavier': xavier, 'glorot' : xavier,
                    'he': he, 'lecun': lecun}

    return tf.cast(function_map[name.lower()](shape), dtype=tf.float32)

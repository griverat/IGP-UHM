import tensorflow as tf

def ECNet_keras(shape, nleads):
    inputs = tf.keras.Input(shape=shape, name="input_data")

    def conv_set(X, n_feat, k_size, act, stride=(3,1,1)):

        conv = tf.keras.layers.Conv3D(
            n_feat,
            k_size,
            activation=act,
            padding="same",
            strides=stride,
            data_format="channels_first",
        )(X)

        return conv

    def dense_set(X, n_feat, act, name):

        dense = tf.keras.layers.Dense(n_feat, activation=act, name=name)(X)

        return dense

    def max_pool(X):

        pool = tf.keras.layers.MaxPool3D(
            (1, 2, 2), strides=2, padding="same", data_format="channels_first",
        )(X)

        return pool

    # conv 1
    conv1 = conv_set(inputs, 36, [3, 4, 8], "relu")
    # pool1
    pool1 = max_pool(conv1)
    # conv2
    conv2 = conv_set(pool1, 36, [3, 2, 4], "relu")
    # pool2
    pool2 = max_pool(conv2)
    # conv3
    conv3 = conv_set(pool2, 36, [3, 2, 4], "relu")
    # flatten
    flat = tf.keras.layers.Flatten()(conv3)
    # dense 1
    dense1 = dense_set(flat, 50, "relu", name="dense1")
    # dense 2
    dense2 = dense_set(dense1, 50, "relu", name="dense2")    
    
    # output1
    output1 = dense_set(dense2, nleads, None, name="eoutput")
    # output2
    output2 = dense_set(dense2, nleads, None, name="coutput")
    # output3
    output3 = dense_set(dense2, 2, None, name="time")
    # output4
    output4 = dense_set(dense2, 2, "softmax", name="extreme_class")
    # model
    model = tf.keras.Model(inputs=inputs, outputs=[output1, output2, output3, output4])
    return model


def ECNet_keras_noclass(shape, nleads):
    inputs = tf.keras.Input(shape=shape, name="input_data")
    # time_sin = tf.keras.Input(shape=(1,))
    # time_cos = tf.keras.Input(shape=(1,))

    def conv_set(X, n_feat, k_size, act, stride=(3,1,1)):

        conv = tf.keras.layers.Conv3D(
            n_feat,
            k_size,
            activation=act,
            padding="same",
            strides=stride,
            data_format="channels_first",
        )(X)

        return conv

    def dense_set(X, n_feat, act, name):

        dense = tf.keras.layers.Dense(n_feat, activation=act, name=name)(X)

        return dense

    def max_pool(X):

        pool = tf.keras.layers.MaxPool3D(
            (1, 2, 2), strides=2, padding="same", data_format="channels_first"
        )(X)

        return pool

    # conv 1
    conv1 = conv_set(inputs, 36, [3, 4, 8], "relu")
    # conv1_cls = conv_set(inputs, 35, [4, 8], "relu")
    # pool1
    pool1 = max_pool(conv1)
    # pool1_cls = max_pool(conv1_cls)
    # conv2
    conv2 = conv_set(pool1, 36, [3, 2, 4], "relu")
    # conv2_cls = conv_set(pool1_cls, 35, [2, 4], "relu")
    # pool2
    pool2 = max_pool(conv2)
    # pool2_cls = max_pool(conv2_cls)
    # conv3
    conv3 = conv_set(pool2, 36, [3, 2, 4], "relu")
    # conv3_cls = conv_set(pool2_cls, 35, [2, 4], "relu")
    # flatten
    flat = tf.keras.layers.Flatten()(conv3)
    # flat_cls = tf.keras.layers.Flatten()(conv3_cls)
    # concatenate time
    # flat = tf.keras.layers.concatenate([flat, time_cos, time_sin])
    # dense 1
    dense1 = dense_set(flat, 50, "relu", name="dense1")
    # dense1_cls = dense_set(flat_cls, 50, "relu", name="dense1_cls")
    # dense 2
    dense2 = dense_set(dense1, 50, "relu", name="dense2")
    # dense2_cls = dense_set(dense1_cls, 50, "relu", name="dense2_cls")
    
    
    # output1
    output1 = dense_set(dense2, nleads, None, name="eoutput")
    # output2
    output2 = dense_set(dense2, nleads, None, name="coutput")
    # output3
    output3 = dense_set(dense2, 2, None, name="time")
    # model
    model = tf.keras.Model(inputs=inputs, outputs=[output1, output2, output3])
    return model

class CriticalScoreIndex(tf.keras.metrics.Metric):
    def __init__(self, name="csi", **kwargs):
        super(CriticalScoreIndex, self).__init__(name=name, **kwargs)
        self.tp = self.add_weight("tp", initializer="zeros")
        self.fp = self.add_weight("fp", initializer="zeros")
        self.fn = self.add_weight("fn", initializer="zeros")

    def update_state(self, y_true, y_pred, **kwargs):
        # y_true = tf.cast(y_true, tf.bool)
        # y_pred = tf.math.greater(y_pred, 0.5)
        y_true = tf.cast(tf.keras.backend.argmax(y_true, axis=-1), tf.bool)
        y_pred = tf.cast(tf.keras.backend.argmax(y_pred, axis=-1), tf.bool)

        true_p = tf.logical_and(tf.equal(y_true, True), tf.equal(y_pred, True))
        false_p = tf.logical_and(tf.equal(y_true, False), tf.equal(y_pred, True))
        false_n = tf.logical_and(tf.equal(y_true, True), tf.equal(y_pred, False))

        self.tp.assign_add(tf.reduce_sum(tf.cast(true_p, self.dtype)))
        self.fp.assign_add(tf.reduce_sum(tf.cast(false_p, self.dtype)))
        self.fn.assign_add(tf.reduce_sum(tf.cast(false_n, self.dtype)))

    def reset_state(self):
        self.tp.assign(0)
        self.fp.assign(0)
        self.fn.assign(0)

    def result(self):
        return self.tp / (self.tp + self.fp + self.fn)

class CategoricalTruePositives(tf.keras.metrics.Metric):
    def __init__(self, name="tp", **kwargs):
        super(CategoricalTruePositives, self).__init__(name=name, **kwargs)

        self.tp = self.add_weight(name="tp", initializer="zeros")

    def update_state(self, y_true, y_pred, **kwargs):
        y_true = tf.cast(tf.keras.backend.argmax(y_true, axis=-1), tf.bool)
        y_pred = tf.cast(tf.keras.backend.argmax(y_pred, axis=-1), tf.bool)

        true_p = tf.logical_and(tf.equal(y_true, True), tf.equal(y_pred, True))
        self.tp.assign_add(tf.reduce_sum(tf.cast(true_p, self.dtype)))

    def reset_state(self):
        self.tp.assign(0)

    def result(self):
        return self.tp

class CategoricalFalsePositives(tf.keras.metrics.Metric):
    def __init__(self, name="fp", **kwargs):
        super(CategoricalFalsePositives, self).__init__(name=name, **kwargs)

        self.fp = self.add_weight(name="fp", initializer="zeros")

    def update_state(self, y_true, y_pred, **kwargs):
        y_true = tf.cast(tf.keras.backend.argmax(y_true, axis=-1), tf.bool)
        y_pred = tf.cast(tf.keras.backend.argmax(y_pred, axis=-1), tf.bool)

        false_p = tf.logical_and(tf.equal(y_true, False), tf.equal(y_pred, True))
        self.fp.assign_add(tf.reduce_sum(tf.cast(false_p, self.dtype)))

    def reset_state(self):
        self.fp.assign(0)

    def result(self):
        return self.fp

class CategoricalTrueNegatives(tf.keras.metrics.Metric):
    def __init__(self, name="tn", **kwargs):
        super(CategoricalTrueNegatives, self).__init__(name=name, **kwargs)

        self.tn = self.add_weight(name="tn", initializer="zeros")

    def update_state(self, y_true, y_pred, **kwargs):
        y_true = tf.cast(tf.keras.backend.argmax(y_true, axis=-1), tf.bool)
        y_pred = tf.cast(tf.keras.backend.argmax(y_pred, axis=-1), tf.bool)

        true_n = tf.logical_and(tf.equal(y_true, False), tf.equal(y_pred, False))
        self.tn.assign_add(tf.reduce_sum(tf.cast(true_n, self.dtype)))

    def reset_state(self):
        self.tn.assign(0)

    def result(self):
        return self.tn

class CategoricalFalseNegatives(tf.keras.metrics.Metric):
    def __init__(self, name="fn", **kwargs):
        super(CategoricalFalseNegatives, self).__init__(name=name, **kwargs)

        self.fn = self.add_weight(name="fn", initializer="zeros")

    def update_state(self, y_true, y_pred, **kwargs):
        y_true = tf.cast(tf.keras.backend.argmax(y_true, axis=-1), tf.bool)
        y_pred = tf.cast(tf.keras.backend.argmax(y_pred, axis=-1), tf.bool)

        false_n = tf.logical_and(tf.equal(y_true, True), tf.equal(y_pred, False))
        self.fn.assign_add(tf.reduce_sum(tf.cast(false_n, self.dtype)))

    def reset_state(self):
        self.fn.assign(0)

    def result(self):
        return self.fn
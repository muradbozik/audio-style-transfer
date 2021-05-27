import tensorflow as tf

class Losses():
    def __init__(self, delta):
        self.delta = delta

    @staticmethod
    def mae(x, y):
        return tf.reduce_mean(tf.abs(x-y))

    @staticmethod
    def mse(x, y):
        return tf.reduce_mean((x-y)**2)

    @staticmethod
    def loss_travel(sa, sab, sa1, sab1):
        l1 = tf.reduce_mean(((sa-sa1) - (sab-sab1))**2)
        l2 = tf.reduce_mean(tf.reduce_sum(-(tf.nn.l2_normalize(sa-sa1,
                            axis=[-1]) * tf.nn.l2_normalize(sab-sab1, axis=[-1])), axis=-1))
        return l1+l2

    def loss_siamese(self, sa, sa1):
        logits = tf.sqrt(tf.reduce_sum((sa-sa1)**2, axis=-1, keepdims=True))
        return tf.reduce_mean(tf.square(tf.maximum((self.delta - logits), 0)))

    @staticmethod
    def d_loss_f(fake):
        return tf.reduce_mean(tf.maximum(1 + fake, 0))

    @staticmethod
    def d_loss_r(real):
        return tf.reduce_mean(tf.maximum(1 - real, 0))

    @staticmethod
    def g_loss_f(fake):
        return tf.reduce_mean(- fake)

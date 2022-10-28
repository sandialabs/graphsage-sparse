import time
import tensorflow as tf


# TimeCallback
class TimeCallback(tf.keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.times = []

    def on_epoch_begin(self, batch, logs={}):
        self.starttime = time.time()

    def on_epoch_end(self, batch, logs={}):
        self.times.append(time.time() - self.starttime)

# Tensorboard
def get_tboardCallback(log_dir, profile_batch='3,5'):
    return tf.keras.callbacks.TensorBoard(
        log_dir=log_dir, profile_batch=profile_batch)
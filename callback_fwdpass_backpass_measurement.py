import tensorflow as tf
import time

# Custom callback to measure forward pass, backpropagation, and weight update times
class TrainingTimeCallback(tf.keras.callbacks.Callback):    
    def on_epoch_begin(self, epoch, logs=None):
        self.start_time = time.time()
        self.forward_pass_time = 0

        self.batch_start_time = 0
        self.batch_end_time = 0
        self.batch_time = 0

    def on_train_batch_begin(self, batch, logs=None):
        self.batch_start_time = time.time()

    def on_train_batch_end(self, batch, logs=None):
        self.batch_end_time = time.time()
        self.batch_time = self.batch_end_time - self.batch_start_time
        self.forward_pass_time += self.batch_time

    def on_epoch_end(self, epoch, logs=None):
        end_time = time.time()
        epoch_time = end_time - self.start_time
        fwd_pass_percent = round((self.forward_pass_time/epoch_time)*100,2)
        wgt_pass_time = epoch_time - self.forward_pass_time
        wgt_and_bkp_percent = round((wgt_pass_time/epoch_time)*100,2)
        print(f"\n  - Total:\t {round(epoch_time,10)}s")
        print(f"  - Forward:\t {round(self.forward_pass_time,10)}s ({fwd_pass_percent}%)")
        print(f"  - Wgt&BkP:\t {round(wgt_pass_time,10)}s ({wgt_and_bkp_percent}%)")
        
        with tf.profiler.experimental.Profile("logs/profile/"):
            with tf.profiler.experimental.Trace("Backpropagation", step_num=epoch):
                model.evaluate(x_test, y_test, verbose=2)

print("TensorFlow version:", tf.__version__)

mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(10)
])

predictions = model(x_train[:1]).numpy()

tf.nn.softmax(predictions).numpy()

loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

loss_fn(y_train[:1], predictions).numpy()

model.compile(optimizer='adam',
              loss=loss_fn,
              metrics=['accuracy'])

# Add the custom callback to record forward pass, backpropagation, and weight update times

model.fit(x_train, y_train, epochs=5, callbacks=[TrainingTimeCallback()])

model.evaluate(x_test,  y_test, verbose=2)

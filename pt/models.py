import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

class ConvModelMNIST(keras.Model): 
	def __init__(self):
		super(ConvModelMNIST, self).__init__()
		self.c32       = layers.Conv2D(32, kernel_size=(3, 3), activation="relu")
		self.c32_2     = layers.Conv2D(32, kernel_size=(3, 3), activation="relu")
		self.mp        = layers.MaxPooling2D(pool_size=(2, 2))
		self.c64       = layers.Conv2D(64, kernel_size=(3, 3), activation="relu")
		self.c64_2     = layers.Conv2D(64, kernel_size=(3, 3), activation="relu")
		self.flatten   = layers.Flatten()
		self.dropout = layers.Dropout(0.25)
		self.dropout_1 = layers.Dropout(0.1)
		self.fc512     = layers.Dense(512, activation="relu")
		self.fc512_2   = layers.Dense(512, activation="relu")
		self.fc256     = layers.Dense(256, activation="relu")
		self.fc10      = layers.Dense(10, activation="softmax")
	
	@tf.function
	def call(self, inputs, training=False): 
		x = self.c32(inputs)
		x = self.c32_2(x)
		x = self.mp(x)
		x = self.c64(x)
		x = self.c64_2(x)
		x = self.mp(x)
		x = self.flatten(x)
		x = self.fc512(x)
		if training:
			x = self.dropout(x)
		x = self.fc512_2(x)
		if training:
			x = self.dropout(x)
		x = self.fc256(x)
		if training:
			x = self.dropout_1(x)
		outputs = self.fc10(x)
		return outputs

	def train_step(self, data):
		x, y_true = data

		with tf.GradientTape() as tape:
			y_pred = self(x, training=True)
			loss_vals = self.compiled_loss(y_true, y_pred)
		
		# Compute gradients, update weights and metrics
		grads = tape.gradient(loss_vals, self.trainable_variables)
		self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
		self.compiled_metrics.update_state(y_true, y_pred)
		return {m.name: m.result() for m in self.metrics}


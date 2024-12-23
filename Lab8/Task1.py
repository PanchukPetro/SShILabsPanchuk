import tensorflow as tf
import numpy as np

n_samples, batch_size, num_steps = 1000, 100, 20000
# cast to float32 cause tensorflow doesn't support float64
X_data = np.random.uniform(1, 10, (n_samples, 1)).astype(np.float32)
y_data = (2 * X_data + 1 + np.random.normal(0, 2, (n_samples, 1))).astype(np.float32)

# Convert data to TensorFlow datasets
dataset = tf.data.Dataset.from_tensor_slices((X_data, y_data)).shuffle(n_samples).batch(batch_size)

# Define model
class LinearRegressionModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # Use `add_weight` to properly register the variables as part of the model
        self.k = self.add_weight(shape=(1, 1), initializer="random_normal", trainable=True, name="slope")
        self.b = self.add_weight(shape=(1,), initializer="zeros", trainable=True, name="bias")

    def call(self, inputs):
        return tf.matmul(inputs, self.k) + self.b

# Instantiate model and optimizer
model = LinearRegressionModel()
optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)

# Define training loop
@tf.function
def train_step(x_batch, y_batch):
    with tf.GradientTape() as tape:
        y_pred = model(x_batch)
        loss = tf.reduce_mean(tf.square(y_batch - y_pred))  # Mean Squared Error
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss

# Training
print(model.trainable_variables)
for step, (X_batch, y_batch) in enumerate(dataset.repeat(num_steps)):
    loss_value = train_step(X_batch, y_batch)
    if (step + 1) % 100 == 0:
        print(f"Step {step+1}: Loss = {loss_value.numpy():.4f}, k = {model.k.numpy().flatten()[0]:.4f}, b = {model.b.numpy().flatten()[0]:.4f}")
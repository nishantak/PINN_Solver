import tensorflow as tf
import numpy as np
from tensorflow.keras import layers, models, initializers
from keras_tuner import HyperModel, RandomSearch

# Set the floating-point precision
DTYPE = 'float32'
tf.keras.backend.set_floatx(DTYPE)

# Constants
alpha = 1
pi = tf.constant(np.pi, dtype=DTYPE)

def fun_u_0(x):
    return -tf.sin(pi * x)

def pde_residual(model, t, x):
    with tf.GradientTape(persistent=True) as tape:
        tape.watch([t, x])
        u = model(tf.concat([t, x], axis=1))
        u_t = tape.gradient(u, t)
    u_x = tape.gradient(u, x)
    u_xx = tape.gradient(u_x, x)
    del tape
    return u_t - alpha * u_xx

activation = 'tanh'
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

class PINNModel(HyperModel):
    def build(self, hp):
        model = models.Sequential()
        model.add(layers.InputLayer(input_shape=(2,)))  
        for i in range(hp.Int('num_layers', 1, 32)):
            model.add(layers.Dense(
                units=hp.Int('units_' + str(i), 32, 512, step=32),
                activation=activation,
                kernel_initializer=initializers.GlorotUniform()
            ))
        model.add(layers.Dense(1, activation=None, kernel_initializer=initializers.GlorotUniform()))  
        return model

def pinn_loss(model, t, x, t_bc, x_bc, u_bc):
    pde_loss = tf.reduce_mean(tf.square(pde_residual(model, t, x)))
    bc_loss = tf.reduce_mean(tf.square(model(tf.concat([t_bc, x_bc], axis=1)) - u_bc))
    return pde_loss + bc_loss

@tf.function
def train_step(model, t, x, t_bc, x_bc, u_bc):
    with tf.GradientTape() as tape:
        loss = pinn_loss(model, t, x, t_bc, x_bc, u_bc)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss

# Define spatial and temporal boundary conditions
x_bc = np.array([0, 1], dtype=np.float32).reshape(-1, 1)  # Spatial boundaries
t_bc = np.array([0, 0], dtype=np.float32).reshape(-1, 1)  # Corresponding time for these boundaries

# Define boundary condition values at the spatial boundaries
u_bc = np.array([fun_u_0(tf.constant(0.0)).numpy(), fun_u_0(tf.constant(1.0)).numpy()], dtype=np.float32).reshape(-1, 1)

# Define initial conditions
x_initial = np.linspace(0, 1, 100).reshape(-1, 1)  # Spatial grid
t_initial = np.zeros_like(x_initial)  # t=0 for all x
u_initial = fun_u_0(tf.constant(x_initial)).numpy().reshape(-1, 1)  # Initial condition values

# Combine boundary and initial conditions
t_bc_combined = np.vstack([t_bc, t_initial])
x_bc_combined = np.vstack([x_bc, x_initial])
u_bc_combined = np.vstack([u_bc, u_initial])

# Convert to TensorFlow tensors
t_bc_combined = tf.convert_to_tensor(t_bc_combined, dtype=DTYPE)
x_bc_combined = tf.convert_to_tensor(x_bc_combined, dtype=DTYPE)
u_bc_combined = tf.convert_to_tensor(u_bc_combined, dtype=DTYPE)

# Assuming t_train and x_train are defined, example:
t_train = np.linspace(0, 1, 100).reshape(-1, 1).astype(DTYPE)
x_train = np.linspace(0, 1, 100).reshape(-1, 1).astype(DTYPE)

hypermodel = PINNModel()
tuner = RandomSearch(
    hypermodel,
    objective='loss',
    max_trials=100,
    executions_per_trial=1,
    directory='pinn_tuning',
    project_name='pinn_burgers'
)
tuner.search(t_train, None, epochs=100, validation_data=(t_bc_combined, u_bc_combined))
best_model = tuner.get_best_models(num_models=1)[0]
print(best_model)

for epoch in range(1000):
    loss = train_step(best_model, t_train, x_train, t_bc_combined, x_bc_combined, u_bc_combined)
    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss: {loss.numpy()}")

u_pred = best_model(tf.concat([t_train, x_train], axis=1))
print("Final loss:", pinn_loss(best_model, t_train, x_train, t_bc_combined, x_bc_combined, u_bc_combined).numpy())
print("Predictions:", u_pred.numpy())

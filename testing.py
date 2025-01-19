import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import Adam
import tensorflow_probability as tfp
from sklearn.model_selection import train_test_split

tfd = tfp.distributions

# Integrated constraint model
class ConstraintModel(object):
    def __init__(self, feature, count, feature_val, count_val,
                 n_hidden=None, dropout_rate=0.5,
                 l2_reg=0., grid_points=1000):
        self.dtype = 'float32'

        # Training data
        self.count = tf.convert_to_tensor(count, self.dtype)
        self.feature = tf.convert_to_tensor(feature, self.dtype)

        # Validation data
        self.count_val = tf.convert_to_tensor(count_val, self.dtype)
        self.feature_val = tf.convert_to_tensor(feature_val, self.dtype)

        # Grid for numerical integration
        self.grid = np.arange(0.5 / grid_points, 1.,
                              1. / grid_points)[np.newaxis, :]
        self.grid = tf.convert_to_tensor(self.grid, self.dtype)

        # Input for keras model
        feature_input = Input(shape=(feature.shape[1],), dtype=self.dtype)

        # Network without likelihood (keras model)
        self.network = ConstraintModel.build_network(feature.shape[1],
                                                     n_hidden,
                                                     dropout_rate,
                                                     l2_reg,
                                                     self.dtype)
        # Build model for training
        dist = self.network(feature_input)
        self.training_model = Model(inputs=feature_input, outputs=dist)

    def fit(self, model_save_file, learning_rate=0.01, batch_size=64, epochs=10, patience=5):
        callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                    restore_best_weights=True,
                                                    patience=patience)

        model_saver = tf.keras.callbacks.ModelCheckpoint(model_save_file,
                                                         save_best_only=True,
                                                         monitor='val_loss')

        # Compile model
        self.training_model.compile(Adam(learning_rate=learning_rate),
                                    loss=self.nll)

        # Fit model
        print(self.training_model.summary())
        self.history = self.training_model.fit(self.feature, self.count,
                                               batch_size=batch_size,
                                               validation_data=(self.feature_val,
                                                                self.count_val),
                                               epochs=epochs,
                                               callbacks=[callback, model_saver])

        self.best_val_loss = np.min(self.history.history['val_loss'])

    def predict(self, feature, count):
        y = tf.convert_to_tensor(count, self.dtype)

        # Beta prior
        pred_dist = self.training_model(feature)
        prior = pred_dist.log_prob(self.grid)

        # Observed count
        obs_count = y[:, 0][:, tf.newaxis]

        # Expected count without selection
        exp_count = y[:, 1][:, tf.newaxis]

        # Predicted rate with selection
        rate = exp_count * self.grid

        # Partial likelihood without Beta prior
        lik = tfd.Poisson(rate).log_prob(obs_count)

        prob = tf.math.exp(lik + prior)
        prob = prob / tf.math.reduce_sum(prob, axis=1, keepdims=True)
        relative_rate = tf.math.reduce_sum(prob * self.grid, axis=1)
        constraint = 1 - relative_rate
        return constraint.numpy(), 1 - pred_dist.mean().numpy().flatten()

    def nll(self, y_true, y_pred):
        # Modify the loss function to accept y_true and y_pred as inputs
        # Observed count
        obs_count = y_true[:, 0][:, tf.newaxis]

        # Expected count without selection
        exp_count = y_true[:, 1][:, tf.newaxis]

        # Predicted rate with selection
        rate = exp_count * self.grid

        # Beta prior
        prior = y_pred.log_prob(self.grid)

        # Partial likelihood without Beta prior
        lik = tfd.Poisson(rate).log_prob(obs_count)

        ll = tfp.math.reduce_logmeanexp(lik + prior, axis=1)
        return -ll

    @staticmethod
    def beta_dist(params):
        # Beta distribution code
        mean = tf.math.sigmoid(params[:, 0])
        kappa = tf.math.exp(params[:, 1])
        alpha = mean * kappa
        beta = (1. - mean) * kappa
        return tfd.Beta(alpha[:, tf.newaxis],
                        beta[:, tf.newaxis])

    @staticmethod
    def build_network(n_feat, n_hidden, dropout_rate, l2_reg,
                      dtype, activation='relu'):
        # Network architecture code
        inputs = Input(shape=(n_feat,))

        regularizer = tf.keras.regularizers.l2(l2_reg)

        if n_hidden is None:
            params = Dense(2, dtype=dtype)(inputs)
        else:
            hidden = inputs
            for n in n_hidden:
                hidden = Dense(n, activation=activation, dtype=dtype,
                               kernel_regularizer=regularizer)(hidden)
                hidden = Dropout(rate=dropout_rate, dtype=dtype)(hidden)
            params = Dense(2, dtype=dtype)(hidden)

        dist = tfp.layers.DistributionLambda(ConstraintModel.beta_dist)(params)
        return Model(inputs=inputs, outputs=dist, name='betaOutputLayer')


# Read input data from text file
input_file_path = "inputdata.txt"
df = pd.read_csv(input_file_path, sep='\t')
count_data = df.iloc[:, 1:3].values
feature = df.iloc[:, 3:].values

# Split data into training and validation sets
feature_train, feature_val, count_train, count_val = train_test_split(feature,
                                                                      count_data,
                                                                      test_size=0.2)

# Train the model
model = ConstraintModel(feature_train, count_train, feature_val, count_val,
                        n_hidden=[64, 64], dropout_rate=0.5, l2_reg=0.001)
model.fit("model_best.hdf5", learning_rate=0.001, batch_size=64, epochs=50, patience=5)

# Predict on the entire dataset
constraint, constraint_by_feature = model.predict(feature, count_data)

# Output results to files
output_file_path = "output1.txt"
out_data = pd.DataFrame.from_dict({df.columns[0]: df.iloc[:, 0],
                                   'DeepLOF_score': constraint})
out_data.to_csv(output_file_path, index=False, sep='\t')

# Save the trained model
model.training_model.save("trained_model.h5")

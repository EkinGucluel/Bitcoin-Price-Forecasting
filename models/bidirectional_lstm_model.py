# models/lstm_model.py

import pandas as pd
from .base_model import BaseModel
import numpy as np
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input, Bidirectional

class BidirectionalLstmModel(BaseModel):
    """
    A Bidirectional LSTM Forecasting Model.
    """
    def __init__(self, params=None):
        if params is None:
            # sequence_length is the most important hyperparameter!
            params = {
                'sequence_length': 10, # Look at the last 10 days of features
                'lstm_units': 50,
                'dropout_rate': 0.2,
                'epochs': 70,
                'batch_size': 32
            }
        super().__init__("Bidirectional LSTM", params)

    def _create_sequences(self, X, y, sequence_length):
        Xs, ys = [], []
        for i in range(len(X) - sequence_length):
            Xs.append(X.iloc[i:(i + sequence_length)].values)
            ys.append(y.iloc[i + sequence_length])
        return np.array(Xs), np.array(ys)

    def _build_model(self, input_shape):
        model = Sequential()
        model.add(Input(shape=input_shape))
        # The LSTM layer is now WRAPPED by the Bidirectional layer.
        # This automatically creates a forward and backward LSTM and merges their outputs.
        model.add(Bidirectional(LSTM(units=self.params['lstm_units'], activation='relu')))
        model.add(Dropout(self.params['dropout_rate']))

        # The output is now a single neuron to predict one price.
        model.add(Dense(units=1))

        model.compile(optimizer='adam', loss='mean_squared_error')
        return model

    def train(self, X_train, y_train):
        print(f"Training {self.model_name}...")

        # Creates the 3D sequences from the 2D data
        X_train_seq, y_train_seq = self._create_sequences(X_train, y_train, self.params['sequence_length'])

        # Builds the model architecture
        # The input shape is (timesteps, features)
        input_shape = (X_train_seq.shape[1], X_train_seq.shape[2])
        self.model = self._build_model(input_shape)

        print("Model Summary:")
        self.model.summary()

        # Fit the model
        self.model.fit(
            X_train_seq,
            y_train_seq,
            epochs=self.params['epochs'],
            batch_size=self.params['batch_size'],
            verbose=1 # Show the training progress
        )
        print("Training complete.")

    def predict(self, X_test):
        print(f"Predicting with {self.model_name}...")

        # Create a "dummy" or "fake" y series. It's just a series of zeros. it's just a placeholder to satisfy the function's signature.
        dummy_y = pd.Series(np.zeros(len(X_test)))

        # Call the function to create the X sequences and create corresponding y sequences of zeros, which it immediately discards with the underscore _.
        X_test_seq, _ = self._create_sequences(X_test, dummy_y, self.params['sequence_length'])

        # Add a safety check in case X_test is too short to create any sequences.
        if X_test_seq.shape[0] == 0:
            return np.array([]) # Avoids an error

        return self.model.predict(X_test_seq)
import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np

class AutoencoderDetector:
    def __init__(self, encoding_dim: int = 8):
        self.encoding_dim = encoding_dim
        self.model: models.Model | None = None
        self.threshold: float | None = None
        self.X_train: np.ndarray | None = None

    def fit(self, X_train: np.ndarray, epochs: int = 50, batch_size: int = 32, threshold_percentile: float = 95):
        self.X_train = X_train
        input_dim = X_train.shape[1]

        input_layer = layers.Input(shape=(input_dim,))
        encoded = layers.Dense(16, activation='relu')(input_layer)
        encoded = layers.Dense(self.encoding_dim, activation='relu')(encoded)
        decoded = layers.Dense(16, activation='relu')(encoded)
        decoded = layers.Dense(input_dim, activation='linear')(decoded)

        self.model = models.Model(inputs=input_layer, outputs=decoded)
        self.model.compile(optimizer='adam', loss='mse')

        self.model.fit(
            X_train,
            X_train,
            epochs=epochs,
            batch_size=batch_size,
            verbose=0,
            validation_split=0.1
        )

        # Compute threshold from training reconstruction error
        X_pred = self.model.predict(X_train)
        errors = np.mean((X_train - X_pred) ** 2, axis=1)
        self.threshold = np.percentile(errors, threshold_percentile)

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        if self.model is None or self.threshold is None or self.X_train is None:
            raise RuntimeError("Model not trained. Call fit() first.")

        X_pred = self.model.predict(X_test)
        errors = np.mean((X_test - X_pred) ** 2, axis=1)
        return (errors > self.threshold).astype(int)

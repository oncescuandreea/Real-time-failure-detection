from sklearn.metrics import accuracy_score
from tensorflow.keras.models import Sequential
import numpy as np


def score_nn(X: np.ndarray, y: np.ndarray, model: Sequential, number_classes: int):
    """
    Function returning the accuracy score for the artificial neural networks
    Inputs:
        X - features
        y - correct numerical labels
        model - NN model
        number_classes - number of classes
    Outputs:
        accuracy__nn - accuracy score
        predicted_cnn1 - label prediction
    """
    model_predictions = model.predict(X)
    predicted_cnn1 = []
    for row in model_predictions:
        one_hot_row_prediction = []
        for i in range(0, number_classes):
            one_hot_row_prediction.append(0)
        one_hot_row_prediction[np.argmax(row)] = 1
        predicted_cnn1.append(one_hot_row_prediction)
    predicted_cnn1 = np.asarray(predicted_cnn1)
    predicted_cnn1 = predicted_cnn1.astype(np.float32)

    accuracy__nn = accuracy_score(y, predicted_cnn1, normalize=True)  # calculate accuracy score on test data
    return accuracy__nn, predicted_cnn1

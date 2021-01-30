import numpy as np
from sklearn.naive_bayes import GaussianNB
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras import regularizers
from tensorflow.keras import optimizers
import tensorflow
from sklearn import svm
from sklearn.metrics import confusion_matrix
from utils.utils_evaluate import score_nn
from sklearn.metrics import accuracy_score
import _io


def initialise_nn_model(number_classes: int, no_hidden: int, regularizer: float, activation_fct: str, no_layers: int):
    """
    Function creates a ANN model based on the variable given by the user
    Inputs:
        number_classes - number of classes corresponding to number of output nodes
        no_hidden - number of nodes in the hidden layers
        regularizer - value of the regularizer to be used
        activation_fct - activation function for the hidden layers
        no_layers - number of hidden layers
    Outputs:
        model_nn - Neural network model
    """
    model_nn = Sequential()
    np.random.seed(0)
    model_nn.add(
        Dense(no_hidden, input_dim=73, activation=activation_fct, kernel_regularizer=regularizers.l2(regularizer)))
    if no_layers > 1:
        no_layers -= 1
        for i in range(0, no_layers):
            model_nn.add(Dense(no_hidden, activation=activation_fct, kernel_regularizer=regularizers.l2(regularizer)))

    model_nn.add(Dense(number_classes, activation='softmax'))
    return model_nn


def train_nn(no_classes: int, dict_nn: dict, sgd: optimizers.SGD, dict_xy: dict,
             accuracy_nn_test_list: list, callback: tensorflow.keras.callbacks.EarlyStopping,
             accuracy_nn_val_list: list, minmax: dict, conf_matrix: dict, label_type: str):
    nn_model = initialise_nn_model(number_classes=no_classes,
                                   no_hidden=dict_nn['no_hidden'],
                                   regularizer=dict_nn['regularizer'],
                                   activation_fct=dict_nn['activation_fct'],
                                   no_layers=dict_nn['no_layers'])
    nn_model.compile(loss=dict_nn['loss_fct'],
                     optimizer=sgd,
                     metrics=['accuracy'])

    if label_type == 'NLP':
        label = '_NLP'
    else:
        label = ''

    # fit models to training data actual labels
    try:
        history = nn_model.fit(dict_xy["X_train"],
                               dict_xy[f"y_train{label}_cat"],
                               validation_data=(dict_xy["X_val"],
                                                dict_xy[f"y_val{label}_cat"]),
                               epochs=dict_nn['number_of_epochs'],
                               callbacks=[callback],
                               batch_size=1)
    except TypeError:
        return 0, 0, 0
    [score_nn_test, predicted_nn_test] = score_nn(dict_xy["X_test"],
                                                  dict_xy["y_test_cat"],
                                                  nn_model, no_classes)
    accuracy_nn_test_list.append(score_nn_test)

    [score_nn_val, predicted_nn_val] = score_nn(dict_xy["X_val"],
                                                dict_xy[f"y_val{label}_cat"],
                                                nn_model, no_classes)
    accuracy_nn_val_list.append(score_nn_val)

    if score_nn_test >= minmax['max']:
        minmax['max'] = score_nn_test
        conf_matrix['conftestmax'] = \
            confusion_matrix(np.argmax(dict_xy["y_test_cat"], axis=-1),
                             np.argmax(predicted_nn_test, axis=-1),
                             labels=list(range(0, no_classes)))

    if score_nn_test <= minmax['min']:
        minmax['min'] = score_nn_test
        conf_matrix['conftestmin'] = \
            confusion_matrix(np.argmax(dict_xy["y_test_cat"], axis=-1),
                             np.argmax(predicted_nn_test, axis=-1),
                             labels=list(range(0, no_classes)))

    if score_nn_val >= minmax['maxv']:
        minmax['maxv'] = score_nn_val
        conf_matrix['confvalmax'] = \
            confusion_matrix(np.argmax(dict_xy[f"y_val{label}_cat"], axis=-1),
                             np.argmax(predicted_nn_val, axis=-1),
                             labels=list(range(0, no_classes)))

    if score_nn_val <= minmax['minv']:
        minmax['minv'] = score_nn_val
        conf_matrix['confvalmin'] = \
            confusion_matrix(np.argmax(dict_xy[f"y_val{label}_cat"], axis=-1),
                             np.argmax(predicted_nn_val, axis=-1),
                             labels=list(range(0, no_classes)))

    return minmax, conf_matrix, history


def train_svm(dict_svm: dict, dictXy: dict, accuracy_SVM_test_list: list,
              accuracy_SVM_val_list: list, minmax: dict, conf_matrix: dict, label_type: str):
    if label_type == 'NLP':
        label = '_NLP'
    else:
        label = ''

    model_svm = svm.SVC(kernel=dict_svm['kernel'],
                        C=dict_svm['C'],
                        gamma=dict_svm['gamma'],
                        decision_function_shape=dict_svm['decision_function'],
                        class_weight='balanced')

    model_svm.fit(dictXy["X_train"], dictXy[f"y_train{label}"])

    validation = model_svm.predict(dictXy["X_val"])
    score_svm_val = accuracy_score(dictXy[f"y_val{label}"], validation, normalize=True)
    accuracy_SVM_val_list.append(score_svm_val)

    predicted_svm = model_svm.predict(dictXy["X_test"])
    score_svm = accuracy_score(dictXy["y_test"], predicted_svm, normalize=True)
    accuracy_SVM_test_list.append(score_svm)

    if score_svm >= minmax['max']:
        minmax['max'] = score_svm
        conf_matrix['conftestmax'] = confusion_matrix(dictXy["y_test"],
                                                      predicted_svm)
    if score_svm <= minmax['min']:
        minmax['min'] = score_svm
        conf_matrix['conftestmin'] = confusion_matrix(dictXy["y_test"],
                                                      predicted_svm)
    if score_svm_val >= minmax['maxv']:
        minmax['maxv'] = score_svm_val
        conf_matrix['confvalmax'] = confusion_matrix(dictXy[f"y_val{label}"],
                                                     validation)
    if score_svm_val <= minmax['minv']:
        minmax['minv'] = score_svm_val
        conf_matrix['confvalmin'] = confusion_matrix(dictXy[f"y_val{label}"],
                                                     validation)
    return minmax, conf_matrix


def train_nb(X_traintot: np.ndarray, y_traintot: np.ndarray, X_test: np.ndarray, y_test: np.ndarray,
             f: _io.TextIOWrapper, label: str = ''):
    modelG = GaussianNB()
    modelG.fit(X_traintot, y_traintot)
    predictedG = modelG.predict(X_test)
    scoreG = accuracy_score(y_test, predictedG, normalize=True)

    print(f"Accuracy NB {label} is:", file=f)
    print(scoreG, file=f)
    print(f"Confusion matrix for {label} Naive Bayes:", file=f)
    print(confusion_matrix(y_test, predictedG), file=f)

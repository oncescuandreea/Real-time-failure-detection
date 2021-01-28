import random
import numpy as np
import _io


def parameters_nn():
    """
    Function used to perform random search for the neural network hyperparameters
    Outputs:
        randomly chosen hyperparameters
    """
    no_hidden_val = range(1, 4)
    regularizer_val = np.linspace(0.001, 0.1, 1000)
    functions = ['tanh', 'relu', 'sigmoid', 'exponential']
    no_hidden_nodes = range(100, 300)
    learning_rates = np.linspace(0.0001, 0.02, 1000)
    epochs = range(130, 220)

    no_hidden_layers = random.sample(no_hidden_val, 1)[0]
    no_hidden = random.sample(no_hidden_nodes, 1)[0]
    activation_fct = functions[random.randint(0, len(functions) - 1)]
    regularizer = random.sample(list(regularizer_val), 1)[0]
    learning_rate = random.sample(list(learning_rates), 1)[0]
    number_of_epochs = random.sample(epochs, 1)[0]
    return no_hidden, no_hidden_layers, activation_fct, regularizer, learning_rate, number_of_epochs


def parameters_svm():
    """
    Function used to perform random search for the SVM hyperparameters
    Outputs:
        randomly chosen hyperparameters
    """
    kernels = ['linear', 'poly', 'rbf', 'sigmoid']
    kernel = kernels[random.randint(0, len(kernels) - 1)]

    range_powers_C = range(-5, 5)
    power_C = random.sample(range_powers_C, 1)[0]
    C = 10 ** power_C

    decision_function_shapes = ['ovr', 'ovo']
    decision_function = decision_function_shapes[random.randint(0, len(decision_function_shapes) - 1)]

    values_gamma = np.linspace(0, 1, 50)
    gamma_float = random.sample(list(values_gamma), 1)[0]
    gammas = ['auto', 'scale', gamma_float]
    gamma = gammas[random.randint(0, len(gammas) - 1)]
    return kernel, C, gamma, decision_function


def get_parameter_sets(number_of_tests: int):
    """
    Function used to retrieve {number_of_tests} lists of sets of randomly chosen
    parameters for NN and SVM before fixing the random seeds. Used to later
    choose the best parameters
    Inputs:
        number_of_tests - number of sets of hyperparameters to be chosen before
            deciding the parameters
    Outputs:
        list_of_params_nn [number_of_tests x 6] - list of randomly chosen
            hyperparameters for the neural network
        list_of_params_svm [number_of_tests x 4] - list of randomly chosen
            hyperparameters for the SVM
    """
    # NN randomly chosen parameters
    list_of_params_nn = []
    for i in range(0, number_of_tests + 1):
        list_of_params_nn.append(parameters_nn())

    # SVM randomly chosen parameters
    list_of_params_svm = []
    for i in range(0, number_of_tests + 1):
        list_of_params_svm.append(parameters_svm())

    return list_of_params_nn, list_of_params_svm


def get_parameters(random_bool: bool, list_of_params: list, list_of_params_svm: list, number_of_tests: int,
                   file: _io.TextIOWrapper):
    """
    Function returning two dictionaries with the chosen hyperparameters for NN
    and for the SVM. It also prints to the file the values
    Inputs:
        random - True if random search hyperparameter values wanted
                   False to use best hyperparameters found so far
        list_of_params [number_of_tests x 6] - list of lists containing NN
            hyperparameters
        list_of_params_svm [number_of_tests x 4] - list of lists containing SVM
            hyperparameters
        number_of_tests - number of sets of randomly chosen parameters
            corresponding to the number of times the scrips will be run
        file - file to which hyperparameters are written
    Outputs:
        dict_nn - dictionary with hyperparameters for NN
        dict_svm - dictionary with hyperparameters for SVM
    """
    dict_nn = {}
    dict_svm = {}
    if random_bool is False:
        dict_nn['no_hidden'] = 189  # used to be 140 best 180
        dict_nn['no_layers'] = 2  # used to be 2
        dict_nn['activation_fct'] = 'relu'  # used to be relu
        dict_nn['regularizer'] = 0.0019909909909909913  # used to be 0.01
        dict_nn['learning_rate'] = 0.01703193193193193  # used to be 0.01
        dict_nn['number_of_epochs'] = 159  # used to be 130 # 154

        dict_svm['kernel'] = 'sigmoid'  # used to be linear
        dict_svm['C'] = 1000  # used to be 1
        dict_svm['gamma'] = 'auto'  # used to be 1
        dict_svm['decision_function'] = 'ovo'  # as in report
    else:
        dict_nn['no_hidden'] = list_of_params[number_of_tests][0]
        dict_nn['no_layers'] = list_of_params[number_of_tests][1]
        dict_nn['activation_fct'] = list_of_params[number_of_tests][2]
        dict_nn['regularizer'] = list_of_params[number_of_tests][3]
        dict_nn['learning_rate'] = list_of_params[number_of_tests][4]
        dict_nn['number_of_epochs'] = list_of_params[number_of_tests][5]

        dict_svm['kernel'] = list_of_params_svm[number_of_tests][0]
        dict_svm['C'] = list_of_params_svm[number_of_tests][1]
        dict_svm['gamma'] = list_of_params_svm[number_of_tests][2]
        dict_svm['decision_function'] = list_of_params_svm[number_of_tests][3]

    dict_nn['loss_fct'] = 'categorical_crossentropy'  # used to be categorical crossentropy
    dict_nn['decay_set'] = 1e-2 / dict_nn['number_of_epochs']  # used to be 1e-2/number_of_epochs

    print(file=file)
    print("No of hidden nodes: " + str(dict_nn['no_hidden']), file=file)
    print(file=file)
    print("Value of regularizer term " + str(dict_nn['regularizer']), file=file)
    print(file=file)
    print("Activation function is " + dict_nn['activation_fct'], file=file)
    print(file=file)
    print("Learning rate is " + str(dict_nn['learning_rate']), file=file)
    print(file=file)
    print("Momentum not used", file=file)
    print(file=file)
    print("Number of epochs " + str(dict_nn['number_of_epochs']), file=file)
    print(file=file)
    print("Number of hidden layers" + str(dict_nn['no_layers']), file=file)
    print(file=file)
    print("Loss function used is " + dict_nn['loss_fct'], file=file)
    print(file=file)
    print("Decay is " + str(dict_nn['decay_set']), file=file)

    print("------------------------------", file=file)
    print("SVM C is " + str(dict_svm['C']), file=file)
    print(file=file)
    print("SVM kernel is " + dict_svm['kernel'], file=file)
    print(file=file)
    print("SVM gamma is " + str(dict_svm['gamma']), file=file)
    print(file=file)
    print("SVM decision function is " + dict_svm['decision_function'], file=file)

    return dict_nn, dict_svm

from numpy import mean
import _io
import matplotlib.pyplot as plt


def print_file_test(type_NN_SVM: str, type_true_NLP: str, f: _io.TextIOWrapper,
                    minmax: dict, conf_matrix: dict, accuracy_list: list):
    type_dict = {'NN': 'Neural Networks', 'SVM': 'SVM'}

    print(f"Test accuracy for {type_dict[type_NN_SVM]} with {type_true_NLP} labels is max:", file=f)
    print(minmax['max'], file=f)
    print(f"Test confusion matrix for {type_dict[type_NN_SVM]} with {type_true_NLP} labels is max:", file=f)
    print(conf_matrix['conftestmax'], file=f)
    print(f"Test accuracy for {type_dict[type_NN_SVM]} with {type_true_NLP} labels is min:", file=f)
    print(minmax['min'], file=f)
    print(f"Test confusion matrix for {type_dict[type_NN_SVM]} with {type_true_NLP} labels is min:", file=f)
    print(conf_matrix['conftestmin'], file=f)
    print(f"Mean test accuracy for {type_dict[type_NN_SVM]} with {type_true_NLP} for 100 runs", file=f)
    print(mean(accuracy_list), file=f)


def print_file_val(type_NN_SVM: str, type_true_NLP: str, f: _io.TextIOWrapper,
                   minmax: dict, conf_matrix: dict, accuracy_list: list):
    type_dict = {'NN': 'Neural Networks', 'SVM': 'SVM'}

    print(f"Validation accuracy for {type_dict[type_NN_SVM]} with {type_true_NLP} labels is max:",
          file=f)
    print(minmax['maxv'], file=f)
    print(f"Validation confusion matrix for {type_dict[type_NN_SVM]} with {type_true_NLP} labels is max:",
          file=f)
    print(conf_matrix['confvalmax'], file=f)
    print(f"Validation accuracy for {type_dict[type_NN_SVM]} with {type_true_NLP} labels is min:",
          file=f)
    print(minmax['minv'], file=f)
    print(f"Validation confusion matrix for {type_dict[type_NN_SVM]} with {type_true_NLP} labels is min:",
          file=f)
    print(conf_matrix['confvalmin'], file=f)
    print(f"Mean validation accuracy for {type_dict[type_NN_SVM]} with {type_true_NLP} for 100 runs", file=f)
    print(mean(accuracy_list), file=f)


def print_summary_of_results(f: _io.TextIOWrapper, minmax_nn: dict, conf_matrix_nn: dict,
                             accuracy_nn_test_list: list, minmax_nn_nlp: dict, conf_matrix_nn_nlp: dict,
                             accuracy_nn_test_list_nlp: list, accuracy_nn_val_list: list,
                             accuracy_nn_val_list_nlp: list, minmax_svm: dict, conf_matrix_svm: dict,
                             accuracy_svm_test_list: list, minmax_svm_nlp: dict, conf_matrix_svm_nlp: dict,
                             accuracy_svm_test_list_nlp: list, accuracy_svm_val_list: list,
                             accuracy_svm_val_list_nlp: list):
    print_file_test('NN', 'real', f, minmax_nn, conf_matrix_nn,
                    accuracy_nn_test_list)
    print_file_test('NN', 'NLP', f, minmax_nn_nlp, conf_matrix_nn_nlp,
                    accuracy_nn_test_list_nlp)

    print(".............................", file=f)

    print_file_val('NN', 'real', f, minmax_nn, conf_matrix_nn,
                   accuracy_nn_val_list)
    print_file_val('NN', 'NLP', f, minmax_nn_nlp, conf_matrix_nn_nlp,
                   accuracy_nn_val_list_nlp)

    print(".............................", file=f)

    # =============================================================================

    print_file_test('SVM', 'real', f, minmax_svm, conf_matrix_svm,
                    accuracy_svm_test_list)
    print_file_test('SVM', 'NLP', f, minmax_svm_nlp, conf_matrix_svm_nlp,
                    accuracy_svm_test_list_nlp)

    print(".............................", file=f)

    print_file_val('SVM', 'real', f, minmax_svm, conf_matrix_svm,
                   accuracy_svm_val_list)
    print_file_val('SVM', 'NLP', f, minmax_svm_nlp, conf_matrix_svm_nlp,
                   accuracy_svm_val_list_nlp)

    f.close()

def plot_ml_results(history_nlp, history, accuracy_nn_val_list: list, accuracy_nn_test_list: list,
                    accuracy_svm_val_list: list, accuracy_svm_test_list: list,
                    accuracy_svm_val_list_nlp: list, accuracy_svm_test_list_nlp: list, newdir: str):
    # =============================================================================
    # plot accuracies for train and validation
    # =============================================================================
    plt.figure(1)  # added line compared to previous laptop
    # =============================================================================
    plt.plot(history.history['accuracy'])  # before it was accuracy/acc in between quotes
    plt.plot(history.history['val_accuracy'])  # before it was val_accuracy in between quotes)
    plt.title('Accuracy on train and validation data')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.savefig(newdir + "/NN_Accuracy_train_val.png", dpi=1200)
    # plt.show()

    # =============================================================================
    plt.figure(2)  # added line compared to previous laptop
    # =============================================================================
    plt.plot(history_nlp.history['accuracy'])  # accuracy in between quotes
    plt.plot(history_nlp.history['val_accuracy'])  # before it was val_accuracy in between quotes)
    plt.title('NLP Accuracy on train and validation data')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.savefig(newdir + "/NN_NLP_Accuracy_train_val.png", dpi=1200)
    # plt.show()

    # plot loss for train and validation
    # =============================================================================
    plt.figure(3)  # added line compared to previous laptop
    # =============================================================================
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Loss function value for train and validation data')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper right')
    plt.savefig(newdir + "/NN_Loss_train_val.png", dpi=1200)
    # plt.show()

    # =============================================================================
    plt.figure(4)  # added line compared to previous laptop
    # =============================================================================
    plt.plot(history_nlp.history['loss'])
    plt.plot(history_nlp.history['val_loss'])
    plt.title('NLPLoss function value for train and validation data')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper right')
    plt.savefig(newdir + "/NN_NLP_Loss_train_val.png", dpi=1200)
    # plt.show()

    # =============================================================================
    plt.figure(5)  # added line compared to previous laptop
    # =============================================================================

    plt.plot(accuracy_nn_val_list)
    plt.plot(accuracy_nn_test_list)
    plt.title('NN accuracy on validation and test data given real labels')
    plt.ylabel('Accuracy')
    plt.xlabel('Number of runs')
    plt.legend(['validation', 'test'], loc='upper left')
    plt.savefig(newdir + "/NN_Validation_vs_test.png", dpi=1200)
    # plt.show()

    plt.plot(accuracy_svm_val_list)
    plt.plot(accuracy_svm_test_list)
    plt.title('SVM accuracy on validation and test data')
    plt.ylabel('Accuracy')
    plt.xlabel('Number of runs')
    plt.legend(['validation', 'test'], loc='upper left')
    plt.savefig(newdir + "/SVM_Validation_vs_test.png", dpi=1200)
    # plt.show()

    # =============================================================================
    plt.figure(6)
    # =============================================================================
    plt.plot(accuracy_svm_val_list_nlp)
    plt.plot(accuracy_svm_test_list_nlp)
    plt.title('NLP SVM accuracy on validation and test data')
    plt.ylabel('Accuracy')
    plt.xlabel('Number of runs')
    plt.legend(['validation', 'test'], loc='upper left')
    plt.savefig(newdir + "/SVM_NLP_Validation_vs_test.png", dpi=1200)
    # plt.show()
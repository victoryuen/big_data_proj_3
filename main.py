from data_extraction import *
from model_training import *
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn import tree, svm
import matplotlib.pyplot as plt

def main():
    print("----- Part 3 -----\n")

    # Decision Tree, no features removed
    train_evaluate_print([], DecisionTreeClassifier)

    print("\n----- Part 4 -----\n")

    # SVM, RBF kernel, no features removed
    train_evaluate_print([], svm.SVC, "rbf")

    print("\n----- Part 6 -----\n")

    # Perform Random Forest to get ranking of features least to most important
    # see https://scikit-learn.org/stable/auto_examples/ensemble/plot_forest_importances.html

    # Decision Tree, 1 feature of lowest importance removed
    # train_evaluate_print([lowest 1], DecisionTreeClassifier)

    # Decision Tree, 4 feature of lowest importance removed
    # train_evaluate_print([lowest 4], DecisionTreeClassifier)

    # Decision Tree, 10 feature of lowest importance removed
    # train_evaluate_print([lowest 10], DecisionTreeClassifier)


def train_evaluate_print(excluded_features: list, classifier: any, kernel: str = ""):
    data = get_data(excluded_features);

    if kernel == "":
        trained = train_model(data, classifier)
    else:
        trained = train_model(data, classifier, kernel)

    print("Elapsed time:", trained["time"], "sec")
    
    report = evaluate_model(data, trained["model"])

    print("Accuracy:", report["accuracy"])
    print("Sensitivity:", report["sensitivity"])
    print("Specificity:", report["specificity"])

    if classifier == DecisionTreeClassifier:
        tree.plot_tree(trained["model"], fontsize=6, feature_names=data["features_list"], class_names=["B", "M"])
        plt.show()

    cm_display = ConfusionMatrixDisplay(report["confusion matrix"], display_labels=["B", "M"])
    cm_display.plot()
    plt.show()

if __name__ == '__main__': 
    main()
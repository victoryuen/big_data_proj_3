from data_extraction import *
from model_training import *
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn import tree, svm
import matplotlib.pyplot as plt

def main():
    print("----- Part 3 -----\n")

    # Decision Tree, no features removed
    train_evaluate_print([], DecisionTreeClassifier, True)

    print("\n----- Part 4 -----\n")

    # SVM, RBF kernel, no features removed
    train_evaluate_print([], svm.SVC, True, "rbf")

    print("\n----- Part 6 -----\n")
    # Random forest method
    importance = get_feature_importance()

    # plot graph of top 2 importance
    print("Displaying graph...\n")

    # print(importance[:1])

    # Decision Tree, 1 feature of lowest importance removed
    print("\n----- Lowest 1 removed -----\n")
    train_evaluate_print(importance[:1], DecisionTreeClassifier, False)

    # Decision Tree, 4 feature of lowest importance removed
    print("\n----- Lowest 4 removed -----\n")
    train_evaluate_print(importance[:4], DecisionTreeClassifier, False)

    # Decision Tree, 10 feature of lowest importance removed
    print("\n----- Lowest 10 removed -----\n")
    train_evaluate_print(importance[:10], DecisionTreeClassifier, False)


def train_evaluate_print(excluded_features: list, classifier: any, show_matrix: bool, kernel: str = ""):
    data = get_data(excluded_features)

    if kernel == "":
        trained = train_model(data, classifier)
    else:
        trained = train_model(data, classifier, kernel)

    print("Elapsed time:", trained["time"], "sec")
    
    report = evaluate_model(data, trained["model"])

    print("Accuracy:", report["accuracy"])
    print("Sensitivity:", report["sensitivity"])
    print("Specificity:", report["specificity"], "\n")

    if classifier == DecisionTreeClassifier:
        print("Displaying tree...\n")
        tree.plot_tree(trained["model"], fontsize=6, feature_names=data["features_list"], class_names=["B", "M"])
        plt.show()

    if show_matrix:
        print("Displaying confusion matrix...\n")
        cm_display = ConfusionMatrixDisplay(report["confusion matrix"], display_labels=["B", "M"])
        cm_display.plot()
        plt.show()

if __name__ == '__main__': 
    main()
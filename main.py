from data_extraction import *
from model_training import *
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn import tree, svm
import matplotlib.pyplot as plt

def main():
    print("----- Part 3 -----\n")

    data_all_features = get_data();
    all_features_tree = train_model(data_all_features, DecisionTreeClassifier)

    print("Elapsed time:", all_features_tree["time"], "sec")
    
    all_features_tree_report = evaluate_model(data_all_features, all_features_tree["model"])

    print("Accuracy:", all_features_tree_report["accuracy"])
    print("Sensitivity:", all_features_tree_report["sensitivity"])
    print("Specificity:", all_features_tree_report["specificity"])

    tree.plot_tree(all_features_tree["model"], fontsize=6, feature_names=data_all_features["features_list"], class_names=["B", "M"])
    plt.show()

    all_features_tree_cm = ConfusionMatrixDisplay(all_features_tree_report["confusion matrix"], display_labels=["B", "M"])
    all_features_tree_cm.plot()
    plt.show()

    print("\n----- Part 4 -----\n")

    all_features_svm = train_model(data_all_features, svm.SVC, "rbf")

    print("elapsed time:", all_features_svm["time"], "sec")
    
    # "visualize the confusion matrix"
    # see https://scikit-learn.org/stable/modules/generated/sklearn.metrics.ConfusionMatrixDisplay.html

if __name__ == '__main__': 
    main()
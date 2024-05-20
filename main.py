from data_extraction import *
from model_training import *
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn import tree, svm
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt

def main():
    print("----- Part 3 -----\n")

    # Decision Tree, no features removed
    train_evaluate_print([], DecisionTreeClassifier, True)

    print("\n----- Part 4 -----\n")

    # SVM, RBF kernel, no features removed
    train_evaluate_print([], svm.SVC, True, "rbf")

    print("\n----- Part 6 -----\n")
    rain_forest_features()
    # Perform Random Forest to get ranking of features least to most important
    # see https://scikit-learn.org/stable/auto_examples/ensemble/plot_forest_importances.html

    # Decision Tree, 1 feature of lowest importance removed
    print("\n----- Lowest 1 removed -----\n")
    # train_evaluate_print([lowest 1], DecisionTreeClassifier, False)

    # Decision Tree, 4 feature of lowest importance removed
    print("\n----- Lowest 4 removed -----\n")
    # train_evaluate_print([lowest 4], DecisionTreeClassifier, False)

    # Decision Tree, 10 feature of lowest importance removed
    print("\n----- Lowest 10 removed -----\n")
    # train_evaluate_print([lowest 10], DecisionTreeClassifier, False)


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
    print("Specificity:", report["specificity"])

    if classifier == DecisionTreeClassifier:
        tree.plot_tree(trained["model"], fontsize=6, feature_names=data["features_list"], class_names=["B", "M"])
        plt.show()

    if show_matrix:
        cm_display = ConfusionMatrixDisplay(report["confusion matrix"], display_labels=["B", "M"])
        cm_display.plot()
        plt.show()
def rain_forest_features():
    data = get_data([])
    rain_forest_model = RandomForestClassifier(random_state=0)
    rain_forest_model.fit(data["X_train"],data["y_train"])
    important_features = pd.DataFrame(rain_forest_model.feature_importances_, index = data["X_dataframe"].columns).sort_values(by=[0],ascending=True) # top features by lowest importance
    #not sure how to index the dataframe for just the left side by that is the lowest features
    worst_features = []
    # print(len(important_features))
    # for i in range(len(important_features)):
    #     worst_features.append()



if __name__ == '__main__': 
    main()
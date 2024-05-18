
from data_extraction import *
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix,classification_report

def main():
    print("----- Part 3 -----\n")

    data_all_features = get_data();


    clf = DecisionTreeClassifier() 
    clf.fit(data_all_features["X_train"], data_all_features["y_train"])
    y_pred = clf.predict(data_all_features["X_test"])

    print(confusion_matrix(data_all_features["y_test"], y_pred))
    print(classification_report(data_all_features["y_test"], y_pred))
if __name__ == '__main__': 
    main()
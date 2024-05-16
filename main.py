
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix,classification_report
def main():
    breast_cancer_df = pd.read_csv("breast-cancer.csv") # reading
    new_df = breast_cancer_df.dropna() #data cleaning 
    X = new_df.iloc[:,2:] # data with all features
    y = new_df.iloc[:,1] # class / column that we want
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2) # split data 80/20
    clf = DecisionTreeClassifier() 
    clf.fit(X_train,y_train)
    y_pred = clf.predict(X_test)
    print(confusion_matrix(y_test,y_pred))
    print(classification_report(y_test,y_pred))
if __name__ == '__main__': 
    main()
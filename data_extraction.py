import pandas as pd
from sklearn.model_selection import train_test_split

# Part 1
breast_cancer_df = pd.read_csv("breast-cancer.csv") # reading

# Part 2
clean_df = breast_cancer_df.dropna() # data cleaning 

def get_data(columns_to_drop:list = []) -> dict[str, list]:
    """
    Returns the testing X and y,
    and training X and y, 
    at 80/20 split, 
    excluding the features listed in columns_to_drop
    """
    new_df = clean_df.drop(columns_to_drop, axis=1)

    X = new_df.iloc[:,2:] # columns with all features
    y = new_df.iloc[:,1] # diagnosis column

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2) # split data 80/20

    features_list = X.columns.tolist()

    return {"X_train": X_train, "X_test": X_test, "y_train": y_train, "y_test": y_test, "features_list": features_list, "X_dataframe" :X}
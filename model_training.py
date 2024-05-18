import time
from sklearn.metrics import confusion_matrix, classification_report

def train_and_evaluate_model(data: dict[str, list], classifier: any, kernel: str = "") -> dict[str, any]:
    """
    From the given data, train a model of the given classifier.
    Returns the model itself and evaluatory metrics, 
    like elapsed time for training, accuracy, sensitivity, specificity.
    confusion matrix
    """

    # train

    if kernel == "":
        clf = classifier();
    else:
        clf = classifier(kernel=kernel)

    start_time = time.time()
    clf.fit(data["X_train"], data["y_train"])
    elapsed_time = time.time() - start_time

    # evaluate

    y_pred = clf.predict(data["X_test"])
    confusion = confusion_matrix(data["y_test"], y_pred)
    report = classification_report(data["y_test"], y_pred)

    return {"model": clf, "confusion matrix": confusion, "report": report, "time": elapsed_time}
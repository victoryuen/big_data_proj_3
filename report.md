# CSCI 493.76: Big Data Technologies

# Project 3

# Steven Hsui, Victor Yuen

## Part 1

### Question

(Readin file) Goto the following website to download the Breast Cancer Dataset. Read the CSV file into panda data frame. You can choose any data structure type to store the values (pandas Or Mat-Lab). 32 features (or columns).

### Output

(N/A)

## Part 2

### Question

 (data cleaning / Preparation) Remove any row which contain empty cell(s) ”Bad Data.” Split the dataset into training set (80%) and testing (20%)


### Output

(N/A)

## Part 3

### Question

(Modeling and Evaluations) Train the dataset on the Decision Tree Classifier using the training set (Track the training time). Draw the decision Tree. Evaluate your trained model using the testing data. How well does your model perform? Use performance metrics, like accuracy, sensitivity and specificity (recall). Visualize the confusion Matrix.

### Output

#### Terminal
```
Elapsed time: 0.00751042366027832 sec
Accuracy: 0.9035087719298246
Sensitivity: 0.8292682926829268
Specificity: 0.9452054794520548
```

#### Tree
![alt text](pics/tree%20all.png)

#### Confusion Matrix
![alt text](pics/tree%20matrix%20all.png)

## Part 4

### Question

 (Modeling and Evaluations) Train the dataset on the Support Vector Machine (RBF) Classifier (Track the training time). Evaluate your trained model using the testing data. How well does your model perform? Use performance metrics, like accuracy, sensitivity and specificity (recall). Visualize the confusion Matrix

### Output

#### Terminal
```
Elapsed time: 0.002118825912475586 sec
Accuracy: 0.8859649122807017
Sensitivity: 0.7916666666666666
Specificity: 0.9545454545454546
```

#### Confusion Matrix
![alt text](<pics/svm matrix all.png>)

## Part 6

### Question

Find the feature importance using Random Forest Method (link provided in reference section)
- Visualize the top two columns (feature) in x-y coordinate system.
- Remove the feature with the lowest importance and retrain your model using Decision Tree. Draw Decision and Evaluate performance of the model. Track the training time.
- Remove the four features with the lowest importances and retrain your model using Decision Tree. Draw Decision tree and Evaluate performance of the model. Track the training time. 
- Remove the ten features with the lowest importances and retrain your model using Decision Tree. Draw Decision tree and Evaluate performance of the model. Track the training time.

### Output

#### Top two features
![alt text](<pics/top two.png>)

#### Trees

##### 1 removed
```
Elapsed time: 0.009986162185668945 sec
Accuracy: 0.8596491228070176
Sensitivity: 0.8809523809523809
Specificity: 0.8472222222222222
```
![alt text](<pics/tree 1 removed.png>)

##### 4 removed
```
Elapsed time: 0.007499217987060547 sec
Accuracy: 0.9035087719298246
Sensitivity: 0.868421052631579
Specificity: 0.9210526315789473
```
![alt text](<pics/tree 4 removed.png>)

##### 10 removed
```
Elapsed time: 0.006588459014892578 sec
Accuracy: 0.9298245614035088
Sensitivity: 0.8936170212765957
Specificity: 0.9552238805970149
```
![alt text](<pics/tree 10 removed.png>)

## Part 7

### Question

(Analysis and Discussions)
- Which model (from Q3 to Q6) performed the best?
- Does removing least important features speed up training times?
- Does removing least important features lower performance of your model?
- How does removing less important features relevant to Big Data (extremely large dataset)?

### Response
- The one from Q6 with the lowest 10 importance removed had the best accuracy, sensitivity, specificity for some reason.
- The more features removed, the faster the training times.
- The accuracy of the model and sensitivity and specificity all improved by removing less important features
- By removing less important features, we can focus on the features that have the most impact on the model. Especially with larger sets of data, the cost spent on training models with irrelevant or little use features would be costly. It's a trade-off for how much computing resources you want to spend versus how much accuracy you want.

## References

- Pandas documentation: https://pandas.pydata.org/docs/
- SciKit Learn Documentation: https://scikit-learn.org/stable/user_guide.html
- Chat GPT: Prompt:How do i visualize the top two columns (feature) in x-y coordinate system using random forest method? But also account for specific output column being not a value but a letter "B" or "M"
- https://www.youtube.com/watch?v=YkYpGhsCx4c&t
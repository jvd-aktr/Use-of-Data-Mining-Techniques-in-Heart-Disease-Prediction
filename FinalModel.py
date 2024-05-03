import time
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler as ss
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import confusion_matrix

from sklearn.svm import SVC

from sklearn.naive_bayes import GaussianNB

from sklearn.ensemble import RandomForestClassifier

from sklearn.neural_network import MLPClassifier

from sklearn.metrics import precision_score, recall_score

"""

This part of the code Deals with Data visualization and preprocessing

"""
start_time_of_solution = time.time()

# Reading Data from csv file

df = pd.read_csv('dataset_stage5.csv', header=None)

df = df.iloc[1:, :]

df.info()

df.columns = ['age', 'sex', 'cp', 'trestbps', 'chol',
              'fbs', 'restecg', 'thalach', 'exang',
              'oldpeak', 'slope', 'ca', 'thal', 'Smoke', 'target']

df.isnull().sum()

# drop null value from dataset

df = df.dropna(axis=0)

#  >>>>>>>>>>>>>>>>>>>histogram for each attribute<<<<<<<<<<<<<<<<<<
df = df.apply(pd.to_numeric)

# Showing the datatypes of the dataframe
print(df.dtypes)

# Plotting histogram
df.hist(figsize=(13, 9))

# Mapping values to 0 1 for better classification prediction
df['target'] = df.target.map({0: 0, 1: 1, 2: 1, 3: 1, 4: 1})
df['sex'] = df.sex.map({0: 'female', 1: 'male'})
df['thal'] = df.thal.fillna(df.thal.mean())
df['ca'] = df.ca.fillna(df.ca.mean())
df['sex'] = df.sex.map({'female': 0, 'male': 1})

# >>>>>>>>>>>>>>>>>>>>Normalization of the dataset<<<<<<<<<<<<<<<<<<<<
df_to_normalize = df.copy()

for column in df_to_normalize.columns:
    df_to_normalize[column] = (df_to_normalize[column] - df_to_normalize[column].min()) / df_to_normalize[
        column].abs().max()

# Plotting Histogram for Normalized Dataset
df_to_normalize.hist(figsize=(13, 9))

# >>>>>>>>>>>>>>>>>>>>distribution of target vs age<<<<<<<<<<<<<<<<<<<<<<<<<

sns.set_context("paper", font_scale=1, rc={"font.size": 20, "axes.titlesize": 25, "axes.labelsize": 20})
sns.catplot(kind='count', data=df, x='age', hue='target', order=df['age'].sort_values().unique(), height=7, aspect=12/7)

plt.title('Variation of Age for each target class')
plt.show()

# >>>>>>>>>>>>>>>>>>>>distribution of target vs sex<<<<<<<<<<<<<<<<<<<<<<<<<
sns.set_context("paper", font_scale=1, rc={"font.size": 20, "axes.titlesize": 25, "axes.labelsize": 20})
sns.catplot(kind='count', data=df, x='sex', hue='target', order=df['sex'].sort_values().unique(), height=7, aspect=12/7)
plt.title('Variation of Sex for each target class')
plt.show()

# >>>>>>>>>>>>>>>>>>barchart of age vs sex with hue = target<<<<<<<<<<<<<<<<<<<<<
sns.catplot(kind='bar', data=df, y='age', x='sex', hue='target', height=7, aspect=12/7)
plt.title('Distribution of age vs sex with the target class')
plt.show()

# >>>>>>>>>>>>>>>>>>data preprocessing<<<<<<<<<<<<<<<<<<
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

X_norm = df_to_normalize.iloc[:, :-1].values
y_norm = df_to_normalize.iloc[:, -1].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

X_train_norm, X_test_norm, y_train_norm, y_test_norm = train_test_split(X_norm, y_norm, test_size=0.2, random_state=0)

sc = ss()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

X_train_norm = sc.fit_transform(X_train_norm)
X_test_norm = sc.fit_transform(X_test_norm)

print()
print("Training Set from Original Data")
print()
print(X_train)

print()
print("Training Set Normalized Data")
print()
print(X_train_norm)

# Dictionary to store all the accuracy, precision , recall:

final_accuracy_dict_non_normalized = {}

final_accuracy_dict_normalized = {}

final_precision_dict_non_normalized = {}

final_precision_dict_normalized = {}

final_recall_dict_non_normalized = {}

final_recall_dict_normalized = {}

"""

=> This part of the code performs classification of the dataset.
=> We performed classification using 6 classification techniques, they are: 

1. Decision Tree
2. Logistic Regression
3. Support Vector Machine
4. Naive Bayes
5. Random Forest
6. MLP

=> After performing the classifications we calculated the accuracy using confusion matrix
=> We calculated the precision and recall score

=> We compared each of the classifier's Accuracy, Precision and Recall score 
        

"""

start_time_of_running_all_models = time.time()

start_time_of_decision_tree = time.time()

# >>>>>>>>>>>>>>>>>>>>>Decision Tree<<<<<<<<<<<<<<<<<<<<<


classifier = DecisionTreeClassifier()
classifier.fit(X_train, y_train)

# >>>>>>>>>>>Predicting the Test set results for original data<<<<<<<<<<<<

y_pred = classifier.predict(X_test)

cm_test = confusion_matrix(y_pred, y_test)

y_pred_train = classifier.predict(X_train)
cm_train = confusion_matrix(y_pred_train, y_train)

# >>>>>>>>>>>>>>>>>Precision  Recall  Accuracy calculation<<<<<<<<<<<<<<<<<

decision_tree_class_precision = precision_score(y_test, y_pred)
decision_tree_class_recall = recall_score(y_test, y_pred)

print()

print("Decision Tree classifier precision score for Non Normalized Data: " + str(decision_tree_class_precision))

print("Decision Tree Classifier recall score for Non Normalized Data: " + str(decision_tree_class_recall))

MLP_accuracy_non_norm_training_set = (cm_train[0][0] + cm_train[1][1]) / len(y_train)
MLP_accuracy_non_norm_test_set = (cm_test[0][0] + cm_test[1][1]) / len(y_test)

final_accuracy_dict_non_normalized["Decision Tree"] = (MLP_accuracy_non_norm_test_set)

final_precision_dict_non_normalized["Decision Tree"] = (decision_tree_class_precision)

final_recall_dict_non_normalized["Decision Tree"] = (decision_tree_class_recall)

print()
print('Accuracy for training set for Decision Tree (Not Normalized)= {}'.format(
    (cm_train[0][0] + cm_train[1][1]) / len(y_train)))
print('Accuracy for test set for Decision Tree (Not Normalized) = {}'.format(
    (cm_test[0][0] + cm_test[1][1]) / len(y_test)))

# >>>>>>>>>>>>>>>>>>>>>>>>>>>Predicting Test set results for normalized data<<<<<<<<<<<<<<<<<<<<<<<<<<<


classifier_norm = DecisionTreeClassifier()
classifier_norm.fit(X_train_norm, y_train_norm)

y_pred_norm = classifier_norm.predict(X_test_norm)

cm_test_norm = confusion_matrix(y_pred_norm, y_test_norm)

y_pred_train_norm = classifier_norm.predict(X_train_norm)
cm_train_norm = confusion_matrix(y_pred_train_norm, y_train_norm)

# >>>>>>>>>>>>>>>>>>>Precision , Recall , Accuracy calculation for Normalized Dataset<<<<<<<<<<<<<<<<<<

decision_tree_class_precision_norm = precision_score(y_test_norm, y_pred_norm)
decision_tree_class_recall_norm = recall_score(y_test_norm, y_pred_norm)

print()

print("Decision Tree classifier precision score for  Normalized Data: " + str(decision_tree_class_precision_norm))

print("Decision Tree Classifier recall score for  Normalized Data: " + str(decision_tree_class_recall_norm))

MLP_accuracy_norm_training_set = (cm_train_norm[0][0] + cm_train_norm[1][1]) / len(y_train_norm)
MLP_accuracy_norm_test_set = (cm_test_norm[0][0] + cm_test_norm[1][1]) / len(y_test_norm)

final_accuracy_dict_normalized["Decision Tree"] = (MLP_accuracy_norm_test_set)

final_precision_dict_normalized["Decision Tree"] = (decision_tree_class_precision_norm)

final_recall_dict_normalized["Decision Tree"] = (decision_tree_class_recall_norm)

print()
print('Accuracy for training set for Decision Tree (Normalized)= {}'.format(MLP_accuracy_norm_training_set))
print('Accuracy for test set for Decision Tree (Normalized) = {}'.format(MLP_accuracy_norm_test_set))

end_time_of_decision_tree = time.time()

#  >>>>>>>>>>>>>>>>>>>Logistic Regression<<<<<<<<<<<<<<<<<

classifier = LogisticRegression()
classifier.fit(X_train, y_train)

# >>>>>>>>>>>Predicting the Test set results for original data<<<<<<<<<<<<

y_pred = classifier.predict(X_test)

cm_test = confusion_matrix(y_pred, y_test)

y_pred_train = classifier.predict(X_train)
cm_train = confusion_matrix(y_pred_train, y_train)

# >>>>>>>>>>>>>>>Precision  Recall Accuracy calculation<<<<<<<<<<<<<<<<

logistic_regression_class_precision = precision_score(y_test, y_pred)
logistic_regression_class_recall = recall_score(y_test, y_pred)

print()

print("Logistic Regression classifier precision score for non Normalized Data: " + str(
    logistic_regression_class_precision))

print("Logistic Regression Classifier recall score for non Normalized Data: " + str(logistic_regression_class_recall))

MLP_accuracy_non_norm_training_set = (cm_train[0][0] + cm_train[1][1]) / len(y_train)
MLP_accuracy_non_norm_test_set = (cm_test[0][0] + cm_test[1][1]) / len(y_test)

final_accuracy_dict_non_normalized["Logisitic Regression"] = (
    MLP_accuracy_non_norm_test_set)

final_precision_dict_non_normalized["Logistic Regression"] = (logistic_regression_class_precision)

final_recall_dict_non_normalized["Logistic Regression"] = (logistic_regression_class_recall)

print()
print('Accuracy for training set for Logistic Regression = {}'.format((cm_train[0][0] + cm_train[1][1]) / len(y_train)))
print('Accuracy for test set for Logistic Regression = {}'.format((cm_test[0][0] + cm_test[1][1]) / len(y_test)))

# >>>>>>>>>>>>>>>>>>>>>>>>>>>Predicting Test set results for normalized data<<<<<<<<<<<<<<<<<<<<<<<<<<<

classifier_norm = LogisticRegression()
classifier_norm.fit(X_train_norm, y_train_norm)

y_pred_norm = classifier_norm.predict(X_test_norm)

cm_test_norm = confusion_matrix(y_pred_norm, y_test_norm)

y_pred_train_norm = classifier_norm.predict(X_train_norm)
cm_train_norm = confusion_matrix(y_pred_train_norm, y_train_norm)

# >>>>>>>>>>>>>>>>>>>>>Precision  Recall Accuracy calculation for Normalized Dataset <<<<<<<<<<<<<<<<<<<

logistic_regression_class_precision_norm = precision_score(y_test_norm, y_pred_norm)
logistic_regression_class_recall_norm = recall_score(y_test_norm, y_pred_norm)

print()

print(
    "Logistic Regression classifier precision score for  Normalized Data: " + str(logistic_regression_class_precision))

print("Logistic Regression Classifier recall score for  Normalized Data: " + str(logistic_regression_class_recall))

MLP_accuracy_norm_training_set = (cm_train_norm[0][0] + cm_train_norm[1][1]) / len(y_train_norm)
MLP_accuracy_norm_test_set = (cm_test_norm[0][0] + cm_test_norm[1][1]) / len(y_test_norm)

final_accuracy_dict_normalized["Logisitic Regression"] = (
    MLP_accuracy_norm_test_set)

final_precision_dict_normalized["Logistic Regression"] = (logistic_regression_class_precision_norm)

final_recall_dict_normalized["Logistic Regression"] = (logistic_regression_class_recall_norm)

print()
print('Accuracy for training set for Logistic Regression (Normalized) = {}'.format(
    (cm_train_norm[0][0] + cm_train_norm[1][1]) / len(y_train_norm)))
print('Accuracy for test set for Logistic Regression (Normalized) = {}'.format(
    (cm_test_norm[0][0] + cm_test_norm[1][1]) / len(y_test_norm)))

end_time_of_logisting_reg = time.time()

#  >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>SVM<<<<<<<<<<<<<<<<<<<<<<<<<<<

classifier = SVC(kernel='rbf')
classifier.fit(X_train, y_train)

# >>>>>>>>>>>>>>>>>>Predicting the Test set results<<<<<<<<<<<<<<<<<<
y_pred = classifier.predict(X_test)

cm_test = confusion_matrix(y_pred, y_test)

y_pred_train = classifier.predict(X_train)
cm_train = confusion_matrix(y_pred_train, y_train)

# >>>>>>>>>>>>>>>>>>>>Precision  Recall Accuracy calculation<<<<<<<<<<<<<<<<<<

SVM_class_precision = precision_score(y_test, y_pred)
SVM_class_recall = recall_score(y_test, y_pred)

print()
print("SVM classifier precision score for non Normalized Data: " + str(SVM_class_precision))

print("SVM Classifier recall score for non Normalized Data: " + str(SVM_class_recall))

MLP_accuracy_non_norm_training_set = (cm_train[0][0] + cm_train[1][1]) / len(y_train)
MLP_accuracy_non_norm_test_set = (cm_test[0][0] + cm_test[1][1]) / len(y_test)

final_accuracy_dict_non_normalized["SVM"] = (
    MLP_accuracy_non_norm_test_set)

final_precision_dict_non_normalized["SVM"] = (SVM_class_precision)

final_recall_dict_non_normalized["SVM"] = (SVM_class_recall)

print()
print('Accuracy for training set for svm = {}'.format((cm_train[0][0] + cm_train[1][1]) / len(y_train)))
print('Accuracy for test set for svm = {}'.format((cm_test[0][0] + cm_test[1][1]) / len(y_test)))

# >>>>>>>>>>>>>>>>>>>>Predicting the Test set results for normalized data<<<<<<<<<<<<<<<<<<<<<

classifier_norm = SVC(kernel='rbf')
classifier_norm.fit(X_train_norm, y_train_norm)
y_pred_norm = classifier_norm.predict(X_test_norm)

cm_test_norm = confusion_matrix(y_pred_norm, y_test_norm)

y_pred_train_norm = classifier_norm.predict(X_train_norm)
cm_train_norm = confusion_matrix(y_pred_train_norm, y_train_norm)

# >>>>>>>>>>>>>>>>>>>>>>Precision  Recall Accuracy calculation for Normalized Dataset <<<<<<<<<<<<<<<<<<<<<<<

SVM_class_precision_norm = precision_score(y_test_norm, y_pred_norm)
SVM_class_recall_norm = recall_score(y_test_norm, y_pred_norm)

print()
print("SVM classifier precision score for  Normalized Data: " + str(SVM_class_precision))

print("SVM Classifier recall score for  Normalized Data: " + str(SVM_class_recall))

MLP_accuracy_norm_training_set = (cm_train_norm[0][0] + cm_train_norm[1][1]) / len(y_train_norm)
MLP_accuracy_norm_test_set = (cm_test_norm[0][0] + cm_test_norm[1][1]) / len(y_test_norm)

final_accuracy_dict_normalized["SVM"] = (
    MLP_accuracy_norm_test_set)

final_precision_dict_normalized["SVM"] = (SVM_class_precision_norm)

final_recall_dict_normalized["SVM"] = (SVM_class_recall_norm)

print()
print('Accuracy for training set for svm (Normalized)= {}'.format(MLP_accuracy_norm_training_set))
print('Accuracy for test set for svm (Normalized)= {}'.format(MLP_accuracy_norm_test_set))

end_time_of_svm = time.time()

# >>>>>>>>>>>>>>>>>>>>>>Naive Bayes<<<<<<<<<<<<<<<<<<<<<<<
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

classifier = GaussianNB()
classifier.fit(X_train, y_train)

# >>>>>>>>>>>>>>>>>>Predicting the Test set results<<<<<<<<<<<<<<<<<
y_pred = classifier.predict(X_test)

cm_test = confusion_matrix(y_pred, y_test)

y_pred_train = classifier.predict(X_train)
cm_train = confusion_matrix(y_pred_train, y_train)

# >>>>>>>>>>>>>>>>>>>>>Precision  Recall Accuracy calculation<<<<<<<<<<<<<<<<<

naive_bayes_class_precision = precision_score(y_test, y_pred)
naive_bayes_class_recall = recall_score(y_test, y_pred)

print()
print("Naive Bayes classifier precision score for non Normalized Data: " + str(naive_bayes_class_precision))

print("Naive Bayes Classifier recall score for non Normalized Data: " + str(naive_bayes_class_recall))

MLP_accuracy_non_norm_training_set = (cm_train[0][0] + cm_train[1][1]) / len(y_train)
MLP_accuracy_non_norm_test_set = (cm_test[0][0] + cm_test[1][1]) / len(y_test)

final_accuracy_dict_non_normalized["Naive Bayes"] = (
    MLP_accuracy_non_norm_test_set)

final_precision_dict_non_normalized["Naive Bayes"] = (naive_bayes_class_precision)

final_recall_dict_non_normalized["Naive Bayes"] = (naive_bayes_class_recall)

print()
print('Accuracy for training set for Naive Bayes = {}'.format((cm_train[0][0] + cm_train[1][1]) / len(y_train)))
print('Accuracy for test set for Naive Bayes = {}'.format((cm_test[0][0] + cm_test[1][1]) / len(y_test)))

# >>>>>>>>>>>>>>>>>>>>Predicting the Test set results for normalized data<<<<<<<<<<<<<<<<<<<<

classifier_norm = GaussianNB()
classifier_norm.fit(X_train_norm, y_train_norm)

y_pred_norm = classifier_norm.predict(X_test_norm)

cm_test_norm = confusion_matrix(y_pred_norm, y_test_norm)

y_pred_train_norm = classifier_norm.predict(X_train_norm)
cm_train_norm = confusion_matrix(y_pred_train_norm, y_train_norm)

# >>>>>>>>>>>>>>>>>>>>>>Precision  Recall Accuracy calculation for Normaziled Dataset<<<<<<<<<<<<<<<<<<<<<<<<<<<

naive_bayes_class_precision_norm = precision_score(y_test_norm, y_pred_norm)
naive_bayes_class_recall_norm = recall_score(y_test_norm, y_pred_norm)

print()
print("Naive Bayes classifier precision score for  Normalized Data: " + str(naive_bayes_class_precision_norm))

print("Naive Bayes Classifier recall score for  Normalized Data: " + str(naive_bayes_class_recall_norm))

MLP_accuracy_norm_training_set = (cm_train_norm[0][0] + cm_train_norm[1][1]) / len(y_train_norm)
MLP_accuracy_norm_test_set = (cm_test_norm[0][0] + cm_test_norm[1][1]) / len(y_test_norm)

final_accuracy_dict_normalized["Naive Bayes"] = (
    MLP_accuracy_norm_test_set)

final_precision_dict_normalized["Naive Bayes"] = (naive_bayes_class_precision_norm)

final_recall_dict_normalized["Naive Bayes"] = (naive_bayes_class_recall_norm)

print()
print('Accuracy for training set for Naive Bayes (Normalized) = {}'.format(MLP_accuracy_norm_training_set))
print('Accuracy for test set for Naive Bayes (Normalized)= {}'.format(MLP_accuracy_norm_test_set))

end_time_of_naive_bayes = time.time()

# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>Random Forest<<<<<<<<<<<<<<<<<<<<<<<<<
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

classifier = RandomForestClassifier(n_estimators=10)
classifier.fit(X_train, y_train)

# >>>>>>>>>>>>>>>>>>>>>>>>Predicting the Test set results<<<<<<<<<<<<<<<<<<<<
y_pred = classifier.predict(X_test)

cm_test = confusion_matrix(y_pred, y_test)

y_pred_train = classifier.predict(X_train)
cm_train = confusion_matrix(y_pred_train, y_train)

# >>>>>>>>>>>>>>>>>>>>>>>>>Precision  Recall Accuracy calculation<<<<<<<<<<<<<<<<<<<<<<<<

random_forest_class_precision = precision_score(y_test, y_pred)
random_forest_class_recall = recall_score(y_test, y_pred)

print()
print("Random Forest classifier precision score for non  Normalized Data: " + str(random_forest_class_precision))

print("Random Forest Classifier recall score for non Normalized Data: " + str(random_forest_class_recall))

MLP_accuracy_non_norm_training_set = (cm_train[0][0] + cm_train[1][1]) / len(y_train)
MLP_accuracy_non_norm_test_set = (cm_test[0][0] + cm_test[1][1]) / len(y_test)

final_accuracy_dict_non_normalized["Random forest"] = (
    MLP_accuracy_non_norm_test_set)

final_precision_dict_non_normalized["Random Forest"] = (random_forest_class_precision)

final_recall_dict_non_normalized["Random Forest"] = (random_forest_class_recall)

print()
print('Accuracy for training set for Random Forest = {}'.format((cm_train[0][0] + cm_train[1][1]) / len(y_train)))
print('Accuracy for test set for Random Forest = {}'.format((cm_test[0][0] + cm_test[1][1]) / len(y_test)))

# >>>>>>>>>>>>>>>>>>>>>>>Predicting the Test set results for normalized data<<<<<<<<<<<<<<<<<<<<<<<


classifier_norm = RandomForestClassifier(n_estimators=10)
classifier_norm.fit(X_train_norm, y_train_norm)

y_pred_norm = classifier_norm.predict(X_test_norm)

cm_test_norm = confusion_matrix(y_pred_norm, y_test_norm)

y_pred_train_norm = classifier_norm.predict(X_train_norm)
cm_train_norm = confusion_matrix(y_pred_train_norm, y_train_norm)

# >>>>>>>>>>>>>>>>>>>>>>>Precision  Recall Accuracy calculation for Normalized Dataset<<<<<<<<<<<<<<<<<<<<<<<

random_forest_class_precision_norm = precision_score(y_test_norm, y_pred_norm)
random_forest_class_recall_norm = recall_score(y_test_norm, y_pred_norm)

print()
print("Random Forest classifier precision score for  Normalized Data: " + str(naive_bayes_class_precision_norm))

print("Random Forest Classifier recall score for  Normalized Data: " + str(naive_bayes_class_recall_norm))

MLP_accuracy_norm_training_set = (cm_train_norm[0][0] + cm_train_norm[1][1]) / len(y_train_norm)
MLP_accuracy_norm_test_set = (cm_test_norm[0][0] + cm_test_norm[1][1]) / len(y_test_norm)

final_accuracy_dict_normalized["Random forest"] = (
    MLP_accuracy_norm_test_set)

final_precision_dict_normalized["Random Forest"] = (random_forest_class_precision_norm)

final_recall_dict_normalized["Random Forest"] = (random_forest_class_recall_norm)

print()
print('Accuracy for training set for Random Forest (Normalized)= {}'.format(MLP_accuracy_norm_training_set))
print('Accuracy for test set for Random Forest (Normalized)= {}'.format(MLP_accuracy_norm_test_set))

end_time_of_random_forest = time.time()

# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>MLP Classifier<<<<<<<<<<<<<<<<<<<<<<<<<
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

classifier = MLPClassifier(random_state=1)
classifier.fit(X_train, y_train)

# >>>>>>>>>>>>>>>>>>>>>>>>>Predicting the Test set results<<<<<<<<<<<<<<<<<<<<
y_pred = classifier.predict(X_test)

cm_test = confusion_matrix(y_pred, y_test)

y_pred_train = classifier.predict(X_train)
cm_train = confusion_matrix(y_pred_train, y_train)

# >>>>>>>>>>>>>>>>>>>>>>>>>>>Precision  Recall Accuracy calculation<<<<<<<<<<<<<<<<<<<<<

mlp_class_precision = precision_score(y_test, y_pred)
mlp_class_recall = recall_score(y_test, y_pred)

print()
print("MLP classifier precision score for non  Normalized Data: " + str(mlp_class_precision))

print("MLP Classifier recall score for non Normalized Data: " + str(mlp_class_recall))

MLP_accuracy_non_norm_training_set = (cm_train[0][0] + cm_train[1][1]) / len(y_train)
MLP_accuracy_non_norm_test_set = (cm_test[0][0] + cm_test[1][1]) / len(y_test)

final_accuracy_dict_non_normalized["MLP"] = (
    MLP_accuracy_non_norm_test_set)

final_precision_dict_non_normalized["MLP"] = (mlp_class_precision)

final_recall_dict_non_normalized["MLP"] = (mlp_class_recall)

print()
print('Accuracy for training set for MLP  = {}'.format((cm_train[0][0] + cm_train[1][1]) / len(y_train)))
print('Accuracy for test set for MLP  = {}'.format((cm_test[0][0] + cm_test[1][1]) / len(y_test)))

# >>>>>>>>>>>>>>>>>>>>>>>Predicting the Test set results for normalized data<<<<<<<<<<<<<<<


classifier_norm = MLPClassifier(random_state=1)
classifier_norm.fit(X_train_norm, y_train_norm)

y_pred_norm = classifier_norm.predict(X_test_norm)

cm_test_norm = confusion_matrix(y_pred_norm, y_test_norm)

y_pred_train_norm = classifier_norm.predict(X_train_norm)
cm_train_norm = confusion_matrix(y_pred_train_norm, y_train_norm)

# >>>>>>>>>>>>>>>>>>>>>>>Precision  Recall Accuracy calculation for Normalized Dataset<<<<<<<<<<<<<<<<<<<

mlp_class_precision_norm = precision_score(y_test_norm, y_pred_norm)
mlp_class_recall_norm = recall_score(y_test_norm, y_pred_norm)

print()
print("MLP classifier precision score for   Normalized Data: " + str(mlp_class_precision_norm))

print("MLP Classifier recall score for  Normalized Data: " + str(mlp_class_recall_norm))

MLP_accuracy_norm_training_set = (cm_train_norm[0][0] + cm_train_norm[1][1]) / len(y_train_norm)
MLP_accuracy_norm_test_set = (cm_test_norm[0][0] + cm_test_norm[1][1]) / len(y_test_norm)

final_accuracy_dict_normalized["MLP"] = (
    MLP_accuracy_norm_test_set)
final_precision_dict_normalized["MLP"] = (mlp_class_precision_norm)

final_recall_dict_normalized["MLP"] = (mlp_class_recall_norm)

print()
print('Accuracy for training set for MLP (Normalized)  = {}'.format(MLP_accuracy_norm_training_set))
print('Accuracy for test set for MLP (Normalized)  = {}'.format(MLP_accuracy_norm_test_set))

end_time_of_mlp = time.time()

end_time_of_running_all_models = time.time()

print("Execution Time of Decision Tree :" + str(end_time_of_decision_tree-start_time_of_decision_tree) +" Seconds")
print("Execution Time of Logistic Regression :" + str(end_time_of_logisting_reg-end_time_of_decision_tree) +" Seconds")
print("Execution Time of SVM:" + str(end_time_of_svm-end_time_of_logisting_reg) +" Seconds")
print("Execution Time of Naive Bayes :" + str(end_time_of_naive_bayes-end_time_of_svm) +" Seconds")
print("Execution Time of Random Forest :" + str(end_time_of_random_forest-end_time_of_naive_bayes) +" Seconds")
print("Execution Time of MLP:" + str(end_time_of_mlp-end_time_of_random_forest) +" Seconds")


print("Execution Time of All models :" + str(end_time_of_running_all_models-start_time_of_running_all_models) +" Seconds")

"""

This section code if for plotting accuracy, precision, recall bar chart for comparison

"""
final_accuracy_dict_non_normalized.update(
    (key, value * 100) for key, value in final_accuracy_dict_non_normalized.items())

final_accuracy_dict_normalized.update(
    (key, value * 100) for key, value in final_accuracy_dict_normalized.items())

final_precision_dict_non_normalized.update(
    (key, value * 100) for key, value in final_precision_dict_non_normalized.items())

final_precision_dict_normalized.update(
    (key, value * 100) for key, value in final_precision_dict_normalized.items())

final_recall_dict_non_normalized.update(
    (key, value * 100) for key, value in final_recall_dict_non_normalized.items())

final_recall_dict_normalized.update(
    (key, value * 100) for key, value in final_recall_dict_normalized.items())

models = list(final_accuracy_dict_non_normalized.keys())
accuracy = list(final_accuracy_dict_non_normalized.values())

models_norm = list(final_accuracy_dict_normalized.keys())
accuracy_norm = list(final_accuracy_dict_normalized.values())

precision = list(final_precision_dict_non_normalized.values())
precision_norm = list(final_precision_dict_normalized.values())


recall = list(final_recall_dict_non_normalized.values())
recall_norm = list(final_recall_dict_normalized.values())




plt.figure(figsize=(12, 7))
plt.bar(models, accuracy, color=['red', 'green'])
plt.xlabel("Training Models")
plt.xticks(rotation=45)
plt.ylabel("Accuracy Values in %")
plt.title("Classification Models Accuracy Comparison")
# plt.show()

plt.figure(figsize=(12, 7))
plt.bar(models, precision_norm, color=['red', 'green'])
plt.xlabel("Training Models")
plt.xticks(rotation=45)
plt.ylabel("Precision Values in %")
plt.title("Classification Models Precision Comparison")
# plt.show()

plt.figure(figsize=(12, 7))
plt.bar(models, recall_norm, color=['red', 'green'])
plt.xlabel("Training Models")
plt.xticks(rotation=45)
plt.ylabel("Recall Values in %")
plt.title("Classification Models Recall Comparison")
plt.show()

end_time_of_solution = time.time()

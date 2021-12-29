import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score


def describe_dataset(dataset):
    print(dataset.head(), '\n')
    print(dataset.shape, '\n')
    print(dataset.describe(), '\n')
    print(dataset.isnull().sum(), '\n')


def data_preprocessing(dataset):
    imputer = SimpleImputer(fill_value=np.nan, strategy='most_frequent')
    X = imputer.fit_transform(dataset)
    return X


def split_data(dataset):
    X = data_preprocessing(dataset)
    # X -> features, Y -> label
    X = np.delete(X, -1, axis=1)
    Y = dataset.Outcome
    # Split into training and test set
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=0.7, random_state=42)
    return X_train, X_test, Y_train, Y_test


def knn_algorithm(X_train, Y_train, X_test):
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train, Y_train)
    knn_predict = knn.predict(X_test)
    return knn_predict


def naïve_bayes_algorithm(X_train, Y_train, X_test):
    nb = GaussianNB()
    nb.fit(X_train, Y_train)
    nb_predict  =  nb.predict(X_test)
    return nb_predict


def calc_accuracy(Y_test, algorithm_predict):
    print("accuracy score: %.2f " % accuracy_score(Y_test, algorithm_predict))
    print("precision score: %.2f " % precision_score(Y_test, algorithm_predict))
    print("recall score: %.2f " % recall_score(Y_test, algorithm_predict))
    print("F1 score: %.2f " % f1_score(Y_test, algorithm_predict))


dataset_path = './diabetes.csv'
dataset = pd.read_csv(dataset_path)

print('__ Describe Data __')
describe_dataset(dataset)

X_train, X_test, Y_train, Y_test = split_data(dataset)

knn_predict = knn_algorithm(X_train, Y_train, X_test)

nb_predict = naïve_bayes_algorithm(X_train, Y_train, X_test)

print('\n__ KNN Accurecy __')
calc_accuracy(Y_test, knn_predict)

print('\n__ Naïve Bayes Accurecy __')
calc_accuracy(Y_test, nb_predict)
""" Exploring learning curves for classification of handwritten digits """

import matplotlib.pyplot as plt
import numpy
from sklearn.datasets import *
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LogisticRegression
plt.switch_backend("TkAgg")


def display_digits():
    digits = load_digits()
    fig = plt.figure()
    for i in range(10):
        subplot = fig.add_subplot(5, 2, i+1)
        subplot.matshow(numpy.reshape(digits.data[i], (8, 8)), cmap='gray')

    plt.show()


def train_model():
    data = load_digits()
    model = LogisticRegression(C=10**-1)
    num_trials = 10
    train_percentages = [.05, .1, .15, .2, .25, .3, .35, .4, .45, .5, .55, .6,
                         .65, .7, .75, .8, .85, .9, .95]
    test_accuracies = numpy.zeros(len(train_percentages))
    for j in range(len(train_percentages)):
        train_accuracy = []
        test_accuracy = []
        for i in range(num_trials):
            X_train, X_test, y_train, y_test = train_test_split(data.data,
                                                                data.target,
                                                                train_size=train_percentages[j])
            model.fit(X_train, y_train)
            test_accuracy.append(model.score(X_test, y_test))
        test_accuracies[j] = numpy.mean(test_accuracy)

    plt.plot(train_percentages, test_accuracies)
    plt.xlabel('Percentage of Data Used for Training')
    plt.ylabel('Accuracy on Test Set')
    print("I did get here...")
    plt.show()


if __name__ == "__main__":
    # display_digits()
    train_model()
    # train_model()

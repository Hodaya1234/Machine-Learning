"""
Hodaya Koslowsky
"""

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats
import random
import operator


"""
Assignment:
Assume input x is scalar and there are 3 classes with equal priors. Each
conditional density is normal:
f (x | y = a) = N (2a, 1), a = 1, 2, 3.

Sample 100 points from each class and train a logistic regression based
on this training data. Plot on the same graph the estimated posterior
probability p(y = 1 | x) based on the logistic regression you trained
and the posterior probability based on the true distribution:
p(y = 1 | x) = f (x| y = 1) / ( f (x | y = 1) + f (x | y = 2) + f (x | y = 3) )

Draw the graphs for the range [0, 10]
"""


def draw_points(classes, w, b, sample_range=10.0, num_points=100.0):
    """
    Drawing the real distribution and the estimated one, to compare
    :param classes: a dictionary of class number and the mean corresponding to it
    :param w: the parameter matrix
    :param b: the bias vector
    :param sample_range: the range of points to draw
    :param num_points: number of points to sample in the graphs
    :return: showing a plot on the screen
    """
    mean, mean2, mean3 = classes[0], classes[1], classes[2]
    std = 1
    points = np.arange(0.0, sample_range, sample_range / num_points)
    real_y = []
    for point in points:
        y = scipy.stats.norm(mean, std).pdf(point)
        y2 = scipy.stats.norm(mean2, std).pdf(point)
        y3 = scipy.stats.norm(mean3, std).pdf(point)
        real_y.append(y / (y + y2 + y3))
    est_y = []
    for point in points:
        softmax_result = softmax(np.array(w) * np.array(point) + np.array(b))
        est_y.append(softmax_result[0])
    line_up, = plt.plot(points, real_y, 'bo', label='Computed Distribution')
    line_down, = plt.plot(points, est_y, 'go', label='Estimated Distribution')
    plt.legend(handles=[line_up, line_down])
    plt.show()


def softmax(x):
    """
    compute the softmax of the x vector
    :param x: an input vector
    :return: a vector of the same length as x, with corresponding softmax values
    """
    return np.exp(x) / np.sum(np.exp(x), axis=0)


def create_date(classes, std=1, num_points=100):
    """
    Ceate a training data from the three classes of means
    :param classes: a dictionary of class number and the mean corresponding to it
    :param std: standard deviation of the distributions
    :param num_points: number of points from each class
    :return: a dictionary of example (a point) and a tag (one of the classes)
    """
    data = {}
    for one_class, mean in classes.iteritems():
        points = np.random.normal(mean, std, num_points)
        for point in points:
            data[point] = one_class
    return data


def shuffle_data(data):
    """
    Shuffle the dictionary
    :param data: the original data
    :return: a randomly shuffled version
    """
    keys = list(data.keys())
    random.shuffle(keys)
    shuffled_data = {}
    for key in keys:
        shuffled_data[key] = data[key]
    return shuffled_data


def compare_y(correct, wrong, y, softmax_result):
    """
    Compare the tag and the algorithm's result
    :param correct: the number of correct guesses so far
    :param wrong: the number of wrong guesses so far
    :param y: the correct known tag
    :param softmax_result: the algorithm's result
    :return: the updated numbers for correct and wrong so far
    """
    y_hat, value = max(enumerate(softmax_result), key=operator.itemgetter(1))
    if y_hat == y:
        correct += 1
    else:
        wrong += 1
    return [correct, wrong]


def sgd(data, w, b):
    """
    A stochastic gradient descent algorithm to fit w and b to the examples of the data
    :param data: a dictionary of class number and the mean corresponding to it
    :param w: the parameter matrix
    :param b: the bias vector
    :return: trained versions of w and b
    """
    epoch = 130
    w_etha = 0.2
    b_etha = 0.1
    for time in range(epoch):
        shuffled_data = shuffle_data(data)
        for x, y in shuffled_data.iteritems():
            softmax_result = softmax(np.array(w) * np.array(x) + np.array(b))
            for j in range(w.__len__()):
                if j == y:
                    new_w_j = w[j] - w_etha * x * (softmax_result[j] - 1)
                    new_b_j = b[j] - b_etha * (softmax_result[j] - 1)
                else:
                    new_w_j = w[j] - w_etha * x * softmax_result[j]
                    new_b_j = b[j] - b_etha * softmax_result[j]
                w[j] = new_w_j
                b[j] = new_b_j
    return [w, b]


def test(classes, w, b):
    """
    Test the parameters on new examples
    :param classes:
    :param w: the parameter matrix
    :param b: the bias vector
    :return: percentage of correct guesses by the algorithm
    """
    total_correct = 0
    total_wrong = 0
    for time in range(3):
        correct = 0
        wrong = 0
        test_data = create_date(classes)
        for x, y in test_data.iteritems():
            softmax_result = softmax(np.array(w) * np.array(x) + np.array(b))
            [correct, wrong] = compare_y(correct, wrong, y, softmax_result)
            total_correct += correct
            total_wrong += wrong
        print("***\nTest Set " + str(time + 1) + ":\ncorrect: " + str(correct) + ", wrong: " + str(wrong) +
              ", percent right: " + str(round(100 * float(correct) / float(correct + wrong), 3)))
    print("***\nTotal:\ncorrect: " + str(total_correct) + ", wrong: " + str(total_wrong) +
          ", percent right: " + str(round(100 * float(total_correct) / float(total_correct + total_wrong), 3)))


def main():
    """
    Create a dictionary of classes and means,
    Initialize the vector of parameters w, and the bias b, to zeros
    Create training data
    Use SGD to adjust w and b to the training data
    Test the result
    Draw the plot of the result compared with the known distribution
    :return:
    """
    classes = {0: 2, 1: 4, 2: 6}
    w = [0, 0, 0]
    b = [0, 0, 0]
    data = create_date(classes)
    [w, b] = sgd(data, w, b)
    test(classes, w, b)
    draw_points(classes, w, b)


if __name__ == "__main__":
    main()

# Hodaya Koslowsky 313377673
import sys
import numpy as np

'''
These function find a conjunction that fits the training examples.
 The input is X- binary strings that represent whether the i variable of the string is true or false.
            Y- a single bit of result to the boolean conjunction of the corresponding example in X.
The output is h - the conjunction that fits to all of the examples in the input.
'''


def turn_conjunction_to_string(h):
    # transform the conjunction given as a list of positive and negative
    # integers, to a string conjunction in the form "x1,not(x2),x3 etc.
    h = sorted(h, key=abs)
    result = ""
    for number in h:
        if number > 0:
            result += "x" + str(number) + ","
        else:
            result += "not(x" + str(abs(number)) + "),"
    result = result[:-1]
    return result


def calculate_conjunction(h, x):
    # compute the result of the boolean conjunction h,
    # by assigning the x[i] value inside the conjunction's variable: xi or not(xi)
    for i in range(x.size):
        if (i + 1 in h and x[i] == 0) or ( -1 * (i+1) in h and x[i] ==1):
            return False
    return True


def consistency(X, Y, d):
    # the algorithm for computing the conjunction
    h = [num for num in range(-d, d + 1) if num != 0]  # the all negative hypothesis
    prev_h = h
    for example, tag in zip(X, Y):  # go through all the examples and remove un-fitting variables
        if calculate_conjunction(h, example) == 0 and tag == 1:
            h = [i for i in prev_h if (i < 0 and example[-1*i -1] == 0) or (i > 0 and example[i - 1] == 1)]
            prev_h = h
    return h


def extract_data_from_file(filename):
    # return X - domain set, Y - tags and d - length of each x in X
    training_examples = np.loadtxt(filename).astype(int)
    d = training_examples[0].size - 1
    X = training_examples[:, :d]
    Y = training_examples[:, d]
    return [d, X, Y]


def write_to_file(filename, string):
    # write the result to the output file
    f = open(filename, "w+")
    f.write(string)
    f.close()


def main():
    [d, X, Y] = extract_data_from_file(sys.argv[1])
    h = consistency(X, Y, d)
    stringH = turn_conjunction_to_string(h)
    write_to_file("output.txt", stringH)

if __name__ == "__main__":
    main()

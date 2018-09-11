import numpy as np
from scipy.special import expit
from random import shuffle

max_x_val = 255
layer_size = 150
learning_rate = 0.005
input_size = 784
output_size = 10
epoch = 50
validation_percent = 0.2
alpha = 0.01
batch_size = 10
# drop_out_p = 0.5


def prelu(x, derivative=False):
    c = np.zeros(np.shape(x))
    if derivative:
        c[x <= 0] = alpha
        c[x > 0] = 1
    else:
        c[x > 0] = x[x > 0]
        c[x <= 0] = alpha * x[x <= 0]
    return c


def relu(x, derivative=False):
    if derivative:
        x[x <= 0] = 0
        x[x > 0] = 1
        return x
    else:
        return np.maximum(x, np.zeros(np.shape(x)))


def sigmoid(x, derivative=False):
    if derivative:
        return np.multiply(sigmoid(x), 1 - sigmoid(x))
    return expit(x)


def softmax(x):
    """
    compute the softmax of the x vector
    :param x: an input vector
    :return: a vector of the same length as x, with corresponding softmax values
    """
    return np.exp(x) / np.sum(np.exp(x), axis=0)


def activation_func(x, derivative=False):
    return relu(x, derivative)


def shuffle_x_and_y(x, y):
    new_x = []
    new_y = []
    index = range(len(y))
    shuffle(index)
    for i in index:
        new_x.append(x[i])
        new_y.append(y[i])
    return [new_x, new_y]


def predict(x, params):
    w1, b1, w2, b2 = [params[key] for key in ('w1', 'b1', 'w2', 'b2')]
    x = np.array(x)
    x.shape = (input_size, 1)
    z1 = np.dot(w1, x) + b1
    h1 = activation_func(z1)
    z2 = np.dot(w2, h1) + b2
    h2 = softmax(z2)
    return np.argmax(h2)


def forward(x, y, params):
    w1, b1, w2, b2 = [params[key] for key in ('w1', 'b1', 'w2', 'b2')]
    x = np.array(x)
    x.shape = (input_size, 1)
    z1 = np.dot(w1, x) + b1
    h1 = activation_func(z1)
    z2 = np.dot(w2, h1) + b2
    h2 = softmax(z2)

    y_vec = np.zeros(output_size)
    y_vec[np.int(y)] = 1
    loss = -1 * np.dot(y_vec, np.log(h2))
    ret = {'x': x, 'y_vec': y_vec, 'z1': z1, 'h1': h1, 'z2': z2, 'h2': h2, 'loss': loss}
    for key in params:
        ret[key] = params[key]
    return ret


def backward(forward_result):
    x, y_vec, z1, h1, z2, h2, w1, w2, b1, b2 = [forward_result[key] for key in ('x', 'y_vec', 'z1', 'h1', 'z2',
                                                                                'h2', 'w1', 'w2', 'b1', 'b2')]
    y_vec = np.asarray(y_vec)
    y_vec.shape = (output_size, 1)

    dl_dz2 = np.subtract(h2, y_vec)
    dl_dw2 = np.dot(dl_dz2, np.transpose(h1))
    dl_db2 = dl_dz2
    dl_dh1 = np.dot(np.transpose(w2), dl_dz2)
    dl_dz1 = np.multiply(dl_dh1, activation_func(z1, derivative=True))
    dl_dw1 = np.dot(dl_dz1, np.transpose(x))
    dl_db1 = dl_dz1
    return {'b1': dl_db1, 'w1': dl_dw1, 'b2': dl_db2, 'w2': dl_dw2}


def update_parameters(params, derivatives):
    w1, b1, w2, b2 = [params[key] for key in ('w1', 'b1', 'w2', 'b2')]
    dl_dw1, dl_db1, dl_dw2, dl_db2 = [derivatives[key] for key in ('w1', 'b1', 'w2', 'b2')]
    w1 = np.subtract(w1, np.multiply(learning_rate, dl_dw1))
    b1 = np.subtract(b1, np.multiply(learning_rate, dl_db1))
    w2 = np.subtract(w2, np.multiply(learning_rate, dl_dw2))
    b2 = np.subtract(b2, np.multiply(learning_rate, dl_db2))
    return {'w1': w1, 'b1': b1, 'w2': w2, 'b2': b2}


def train_epoch(x, y, params):
    [x, y] = shuffle_x_and_y(x, y)

    epoch_loss = []
    for example, tag in zip(x, y):
        norm_example = np.divide(example, max_x_val)
        fd_result = forward(norm_example, tag, params)
        derivatives = backward(fd_result)
        params = update_parameters(params, derivatives)
        epoch_loss.append(fd_result['loss'])
    return [params, np.mean(epoch_loss)]


def make_prediction(test_x, params):
    prediction = []
    for x in test_x:
        x = np.divide(x, max_x_val)
        prediction.append(int(predict(x, params)))
    return prediction


def check_validation(x, y, params):
    correct = 0.0
    wrong = 0.0
    for example, tag in zip(x, y):
        norm_example = np.divide(example, max_x_val)
        y_hat = predict(norm_example, params)
        if y_hat == tag:
            correct += 1
        else:
            wrong += 1
    return correct / (correct + wrong)


def create_random_matrix(row, col):
    return np.random.rand(row, col)*0.2 - 0.1


def initialize_parameters():
    param = {"w1": create_random_matrix(layer_size, input_size), "b1": create_random_matrix(layer_size, 1),
             "w2": create_random_matrix(output_size, layer_size), "b2": create_random_matrix(output_size, 1)}
    return param


if __name__ == "__main__":
    data_x = np.loadtxt("train_x")
    data_y = np.loadtxt("train_y")
    test_x = np.loadtxt("test_x")
    data_size = np.size(data_y)
    validation_size = int(data_size * validation_percent)

    data_x, data_y = shuffle_x_and_y(data_x, data_y)
    validation_x = data_x[:validation_size]
    validation_y = data_y[:validation_size]
    train_x = data_x[validation_size:]
    train_y = data_y[validation_size:]

    parameters = initialize_parameters()
    for one_epoch in range(epoch):
        parameters, loss = train_epoch(train_x, train_y, parameters)
        validation_correctness = check_validation(validation_x, validation_y, parameters)
        print("\n" + str(one_epoch))
        print("#loss of the train: " + str(loss))
        print("#validation correct: " + str(validation_correctness))
        if validation_correctness > 0.9:
            break
    y_prediction = make_prediction(test_x, parameters)
    np.savetxt("output.txt", y_prediction, fmt='%d', delimiter='\n')


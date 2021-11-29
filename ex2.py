# Tsahi Elhadad 206214165
import numpy as np
import sys


#   Returns the optimal k in KNN algorithm
def get_k_optimal():
    n = len(list_examples_x)
    k = np.sqrt(n)
    k = round(k + (0.2 * n))
    index_separate = round(n * 0.7)
    train_x_ls = list_examples_x[:index_separate]
    test_ls = list_examples_x[index_separate:]
    train_y_ls = list_examples_y[:index_separate]
    real_y = list_examples_y[index_separate:]
    max_val = 0
    max_k = 0
    for i in range(k):
        i += 1
        count = 0
        _knn = KNN(train_x_ls, train_y_ls, test_ls, i)
        ls_prediction = _knn.predict()
        for j in range(len(ls_prediction)):
            if ls_prediction[j] == real_y[j]:
                count += 1
        t_p_percent = count / len(ls_prediction)
        if max_val < t_p_percent:
            max_val = t_p_percent
            max_k = i

    return max_k

#   KNN algorithm class
class KNN():

    #   Constructor - get list of train examples
    #               - get list of train examples lables
    #               - get list of test examples
    #               - get k for the algorithm
    def __init__(self, ls_train_x, ls_train_y, ls_test_x, k=5):
        #   set unique classes list for lables
        self.classes_list = np.unique(ls_train_y)
        #   set number of features
        self.num_features = len(ls_test_x[0])
        self.ls_train_x = ls_train_x
        self.ls_train_y = ls_train_y
        self.ls_test_x = ls_test_x
        self.k = k
        #   set prediction list for predict function
        self.prediction_ls = []

    #   calculate the euclidean distance between two vectors
    def euclid_distance(self, vec1, vec2):
        dist = np.power(np.linalg.norm(vec1 - vec2), 2)
        return dist

    #   predict the examples by knn algorithm
    def predict(self):

        for x_test in self.ls_test_x:
            #   list of distances between each x_test to all x_train examples
            ls_dist = []
            index_y = 0
            for x_train in self.ls_train_x:
                dist = self.euclid_distance(x_test, x_train)
                #   append the x_train, distance, and the label of x_train
                ls_dist.append((x_train, dist, self.ls_train_y[index_y]))
                index_y += 1
            #   sort by distance
            ls_dist.sort(key=lambda tuple_: tuple_[1])
            #   list of lables of closest examples to x_test
            ls_y = []
            for l in range(self.k):
                ls_y.append(ls_dist[l][2])
            #   get max label that appears
            max_val = np.argmax(np.bincount(ls_y))
            #   add the prediction label to prediction list
            self.prediction_ls.append(max_val)

        return self.prediction_ls

    #   accuracy function to get success percent of prediction (by test_y - list of labels)
    def accuracy(self, test_y):
        counter = 0
        prediction = self.predict()
        index = 0
        for y in test_y:
            if y != prediction[index]:
                counter += 1
            index += 1
        return 1 - float(counter / len(test_y))

#   Preceptrom algorithm class
class Preceptrom():
    #   Constructor - get list of train examples
    #               - get list of train examples lables
    #               - get list of test examples
    #               - get learning rate hyper parameter (default = 1)
    #               - get number of epochs/number of iterations (default = 100)
    def __init__(self, ls_train_x, ls_train_y, ls_test_x, lr=1, n_iter=100):
        #   get list of labels/classes for classification
        self.classes_list = np.unique(ls_train_y)
        # set number of features
        self.num_features = len(ls_test_x[0])
        self.ls_train_x = ls_train_x
        self.ls_train_y = ls_train_y
        self.ls_test_x = ls_test_x
        self.lr = lr
        self.epochs = n_iter
        #   set weight vectors for learning
        self.weight_vectors = np.zeros((len(self.classes_list), self.num_features))
        #   set bias column to 1
        self.weight_vectors[:, self.num_features - 1] = [1 for i in self.weight_vectors[:, self.num_features - 1]]
        self.prediction_ls = []

    #   train the weight vectors by the train_x examples
    def train(self):

        for i in range(self.epochs):
            #   set the learning rate hyper parameter by epochs
            self.lr = 1 / np.sqrt(i + 1)

            for x, y in zip(self.ls_train_x, self.ls_train_y):
                #   get label of the max weight multiplication
                y_max = np.argmax(np.dot(self.weight_vectors, x))
                y = int(y)
                #   if wrong prediction
                if y_max != y:
                    self.weight_vectors[y] = self.weight_vectors[y] + self.lr*x
                    self.weight_vectors[y_max] = self.weight_vectors[y_max] - self.lr * x

    #   predict labels of ls_test_x
    def predict(self):

        for x in self.ls_test_x:
            y = np.argmax(np.dot(self.weight_vectors, x))
            self.prediction_ls.append(y)

        return self.prediction_ls

    #   accuracy function to get success percent of prediction (by test_y - list of labels)
    def accuracy(self, test_y):
        counter = 0
        prediction = self.predict()
        index = 0
        for y in test_y:
            if y != prediction[index]:
                counter += 1
            index += 1
        return 1 - float(counter / len(test_y))

#   Passive Aggressive algorithm class
class PA():
    #   Constructor - get list of train examples
    #               - get list of train examples lables
    #               - get list of test examples
    #               - get taho hyper parameter (default = 1)
    #               - get number of epochs/iterations (default = 100)
    def __init__(self, ls_train_x, ls_train_y, ls_test_x, taho=1, n_iter=100):
        #   get list of labels/classes for classification
        self.classes_list = np.unique(ls_train_y)
        self.num_features = len(ls_test_x[0])
        self.ls_train_x = ls_train_x
        self.ls_train_y = ls_train_y
        self.ls_test_x = ls_test_x
        self.taho = taho
        #   set learning rate for handle epochs
        self.lr = 0.1
        self.epochs = n_iter
        #   set weight vectors for learning
        self.weight_vectors = np.zeros((len(self.classes_list), self.num_features))
        #   set bias column to 1
        self.weight_vectors[:, self.num_features - 1] = [1 for i in self.weight_vectors[:, self.num_features - 1]]
        self.prediction_ls = []

    #   returns loss value by hinge loss function
    def loss_func(self, x, w_y, w_y_max):
        return max(0, 1 - np.dot(w_y, x) + np.dot(w_y_max, x))

    #   train the weight vectors by the train_x examples
    def train(self):
        #   temporary variable for update lr parameter
        temp_update_eta = round(self.epochs / 10)
        
        for i in range(self.epochs):

            #   update the parameter every 1/10 of the amount of iterations
            if i == temp_update_eta:
                temp_update_eta = temp_update_eta + self.epochs / 10
                self.lr = self.lr / 2

            for x, y in zip(self.ls_train_x, self.ls_train_y):
                #   get label of max multiplication weight vector
                y_max = np.argmax(np.dot(self.weight_vectors, x))
                y = int(y)
                #   if wrong prediction
                if y_max != y:
                    loss = self.loss_func(x, self.weight_vectors[y], self.weight_vectors[y_max])
                    #   set the loss
                    self.taho = loss / (np.power(np.linalg.norm(x), 2) * 2)
                    self.weight_vectors[y] = self.weight_vectors[y] + self.taho * x * self.lr
                    self.weight_vectors[y_max] = self.weight_vectors[y_max] - self.taho * x * self.lr

    #   predict labels of ls_test_x
    def predict(self):

        for x in self.ls_test_x:
            y = np.argmax(np.dot(self.weight_vectors, x))
            self.prediction_ls.append(y)

        return self.prediction_ls

    #   accuracy function to get success percent of prediction (by test_y - list of labels)
    def accuracy(self, test_y):
        counter = 0
        prediction = self.predict()
        index = 0
        for y in test_y:
            if y != prediction[index]:
                counter += 1
            index += 1
        return 1 - float(counter / len(test_y))

#   SVM algorithm class
class SVM():
    #   Constructor - get list of train examples
    #               - get list of train examples lables
    #               - get list of test examples
    #               - get lambda hyper parameter (default = 1)
    #               - get number of epochs/iterations (default = 100)
    def __init__(self, ls_train_x, ls_train_y, ls_test_x, _lambda=1, lr=0.01, n_iter=100):
        self.classes_list = np.unique(ls_train_y)
        self.num_features = len(ls_test_x[0])
        self.ls_train_x = ls_train_x
        self.ls_train_y = ls_train_y
        self.ls_test_x = ls_test_x
        self.lr = lr
        self.epochs = n_iter
        self._lambda = _lambda
        #   set weight vectors for learning
        self.weight_vectors = np.zeros((len(self.classes_list), self.num_features))
        #   set bias column to 1
        self.weight_vectors[:, self.num_features - 1] = [1 for i in self.weight_vectors[:, self.num_features - 1]]
        self.prediction_ls = []

    #   returns loss value by hinge loss function
    def loss_func(self, x, w_y, w_y_max):
        return max(0, 1 - np.dot(w_y, x) + np.dot(w_y_max, x))

    #   train the weight vectors by the train_x examples
    def train(self):

        for i in range(self.epochs):
            #   set lambda value
            self._lambda = 1 / (i + 1)
            if self._lambda < 0.0001:
                self._lambda = 0.0001

            for x, y in zip(self.ls_train_x, self.ls_train_y):

                y = int(y)
                vec = np.dot(self.weight_vectors, x)
                #   to avoid choose the right label y as max weight vector
                vec[y] = float('-inf')
                y_max = np.argmax(vec)
                loss = self.loss_func(x, self.weight_vectors[y], self.weight_vectors[y_max])

                #   if wrong prediction
                if loss > 0:
                    self.weight_vectors[y] = (1 - (self._lambda * self.lr)) * self.weight_vectors[y] + self.lr * x
                    self.weight_vectors[y_max] = (1 - (self._lambda * self.lr)) * self.weight_vectors[y_max] - self.lr * x
                    #   for update only the other weight vectors by this solution
                    for j in range(len(self.weight_vectors)):
                        if j != y and j != y_max:
                            self.weight_vectors[j] *= (1 - (self._lambda * self.lr))
                else:   # if right prediction, update all weight vectors
                    self.weight_vectors *= (1 - (self._lambda * self.lr))

    #   predict labels of ls_test_x
    def predict(self):

        for x in self.ls_test_x:
            y = np.argmax(np.dot(self.weight_vectors, x))
            self.prediction_ls.append(y)
        return self.prediction_ls

    #   accuracy function to get success percent of prediction (by test_y - list of labels)
    def accuracy(self, test_y):
        counter = 0
        prediction = self.predict()
        index = 0
        for y in test_y:
            if y != prediction[index]:
                counter += 1
            index += 1
        return 1 - float(counter / len(test_y))

#   MinMax normalization function
def minMaxNormalization(list_train_x):
    num_features = len(list_train_x[0])
    new_train_x = np.zeros((len(list_train_x), num_features))
    for i in range(num_features):
        column = list_train_x[:, i]
        x_max = max(column)
        x_min = min(column)
        new_train_x[:, i] = (list_train_x[:, i] - x_min) / (x_max - x_min)
    return new_train_x

#   ZScore normalization function for train
def zScoreNormalizationForTrain(list_train_x):
    num_features = len(list_train_x[0])
    new_train_x = np.zeros((len(list_train_x), num_features))
    mean_ls = []
    dev_ls = []
    for i in range(num_features):
        column = list_train_x[:, i]
        x_mean = np.mean(column)
        x_std = np.std(column)
        new_train_x[:, i] = (list_train_x[:, i] - x_mean) / x_std
        mean_ls.append(x_mean)
        dev_ls.append(x_std)
    return new_train_x, mean_ls, dev_ls

#   ZScore normalization function for test examples
def zScoreNormalizationForTest(list_train_x, mean_train_ls, dev_train_ls):
    num_features = len(list_train_x[0])
    new_train_x = np.zeros((len(list_train_x), num_features))
    for i in range(num_features):
        column = list_train_x[:, i]
        new_train_x[:, i] = (list_train_x[:, i] - mean_train_ls[i]) / dev_train_ls[i]
    return new_train_x

#   Cross Validation/KFolds function
def crossValidation(train_x_list, train_y_list, k=5, test_index=0):
    #   Number of examples
    size = len(train_x_list)
    test_split_x = []
    test_split_y = []
    train_split_x = list()
    train_split_y = list()
    data_x = np.array_split(train_x_list,k)
    data_y = np.array_split(train_y_list,k)
    for i in range(k):
        if i == test_index:
            test_split_x = data_x[test_index]
            test_split_y = data_y[test_index]
            continue
        for vec in data_x[i]:
            train_split_x.append(vec)
        for label in data_y[i]:
            train_split_y.append(label)

    return train_split_x, train_split_y, test_split_x, test_split_y

#   Main function
def main():

    #   Get arguments of files
    train_x_file, train_y_file, test_x_file, out_file_name = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
    list_examples_x = np.loadtxt(train_x_file, delimiter=",")
    list_examples_y = np.loadtxt(train_y_file, delimiter=",")
    list_test_x = np.loadtxt(test_x_file, delimiter=",")

    #   open output file
    out_file = open(out_file_name, "w")

    #   Shuffle the data   #
    mapIndexPosition = list(zip(list_examples_x, list_examples_y))
    np.random.shuffle(mapIndexPosition)
    list_examples_x, list_examples_y = zip(*mapIndexPosition)
    list_examples_x = np.asarray(list_examples_x)
    list_examples_y = np.asarray(list_examples_y)

    #   normalization the data (train and test) with ZScore for: Preceptrom, SVM,PA algorithms.
    ls_zScore_train_x, mean_train_ls, dev_train_ls = zScoreNormalizationForTrain(list_examples_x)
    ls_zScore_test_x = zScoreNormalizationForTest(list_test_x, mean_train_ls, dev_train_ls)

    #   Adding bias to train examples and test examples
    bias_train = [[1] for i in range(len(list_examples_x))]
    bias_test = [[1] for i in range(len(list_test_x))]
    ls_zScore_train_x = np.append(ls_zScore_train_x, bias_train, axis=1)
    ls_zScore_test_x = np.append(ls_zScore_test_x, bias_test, axis=1)

    #   No need to normalize the data for KNN algorithm.
    ls_examples_x = np.append(list_examples_x, bias_train, axis=1)
    ls_test_x = np.append(list_test_x, bias_test, axis=1)

    #   Set k of KNN algorithm
    k = round(np.sqrt(len(ls_examples_x)))
    #   Set algorithms
    _knn = KNN(ls_examples_x, list_examples_y, ls_test_x, k)
    _preceptron = Preceptrom(ls_zScore_train_x, list_examples_y, ls_zScore_test_x)
    _pa = PA(ls_zScore_train_x, list_examples_y, ls_zScore_test_x)
    _svm = SVM(ls_zScore_train_x, list_examples_y, ls_zScore_test_x)

    #   Train the model with train examples
    _preceptron.train()
    _pa.train()
    _svm.train()

    #   Predict the
    pa_predict = _pa.predict()
    svm_pred = _svm.predict()
    prec_predict = _preceptron.predict()
    _knn_predict = _knn.predict()
    
    for i in range(len(list_test_x)):
        out_file.write(
            f"knn: {_knn_predict[i]}, perceptron: {prec_predict[i]}, svm: {svm_pred[i]}, pa: {pa_predict[i]}\n")

    out_file.close()

    #   Acuracies   #
    # test_y_list = np.loadtxt("test_y.txt", delimiter=",")
    # print(_knn.accuracy(test_y_list))
    # print(_preceptron.accuracy(test_y_list))
    # print(_svm.accuracy(test_y_list))
    # print(_pa.accuracy(test_y_list))

    #   Cross Validation Check  #
    # k = 5
    # sum = 0
    # for i in range(k):
    #     train_x_cv,train_y_cv,test_x_cv,test_y_cv = crossValidation(ls_zScore_train_x, list_examples_y, k, i)
    #     preceptron = Preceptrom(train_x_cv, train_y_cv, test_x_cv)
    #     preceptron.train()
    #     accuracy = preceptron.accuracy(test_y_cv)
    #     sum += accuracy
    # print(sum/k)

if __name__ == '__main__':
    main()

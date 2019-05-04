import numpy as np

class MLP:
    # Hyperparameters initialize
    def __init__(self, hidden_units, minibatch_size, regularization_rate, learning_rate):
        self.hidden_units = hidden_units
        self.minibatch_size = minibatch_size
        self.regularization_rate = regularization_rate
        self.learning_rate = learning_rate

    # ReLu (Rectified Linear Unit) function
    def relu_function(self, matrix_content, matrix_dim_x, matrix_dim_y):
        ret_vector = np.zeros((matrix_dim_x, matrix_dim_y))

        for i in range(matrix_dim_x):
            for j in range(matrix_dim_y):
                ret_vector[i, j] = max(0, matrix_content[i,j])

        return ret_vector

    # the gradient of ReLu (Rectified Linear Unit) function
    def grad_relu(self, matrix_content, matrix_dim_x, matrix_dim_y):
        ret_vector = np.zeros((matrix_dim_x, matrix_dim_y))

        for i in range(matrix_dim_x):
            for j in range(matrix_dim_y):
                if matrix_content[i, j] > 0:
                    ret_vector[i, j] = 1
                else:
                    ret_vector[i, j] = 0

        return ret_vector

    # Softmax fucntion
    def softmax_function(self, vector_content):
        return np.exp(vector_content - np.max(vector_content)) / np.sum(np.exp(vector_content - np.max(vector_content)), axis=0)

    # generator for mini-batch
    def iterate_minibatches(self, inputs, targets, batch_size, shuffle=False):
        assert inputs.shape[0] == targets.shape[0]  # 만약 input / output shape 체크

        if shuffle: # batch shuffle 적용
            indices = np.arange(inputs.shape[0])
            np.random.shuffle(indices)

        for start_idx in range(0, inputs.shape[0] - batch_size + 1, batch_size):
            if shuffle:
                excerpt = indices[start_idx:start_idx + batch_size]
            else:
                excerpt = slice(start_idx, start_idx + batch_size)

            yield inputs[excerpt], targets[excerpt]

    #function to train the MLP model
    def train(self, trainX, trainY, epochs):

        # parameter initialize
        w1_mat = np.random.randn(self.hidden_units, 28*28) *np.sqrt(2./(self.hidden_units+28*28))
        w2_mat = np.random.randn(10, self.hidden_units) *np.sqrt(2./(10+self.hidden_units))
        b1_vec = np.zeros((self.hidden_units, 1))
        b2_vec = np.zeros((10, 1))

        # input / output size reshape
        trainX = np.reshape(trainX, (trainX.shape[0], 28*28))
        trainY = np.reshape(trainY, (trainY.shape[0], 1))

        for num_epochs in range(epochs) :
            if num_epochs % 2 == 0:
                print("Current epoch number : ", num_epochs)

            for batch in self.iterate_minibatches(trainX, trainY, self.minibatch_size, shuffle=True):
                x_batch, y_batch = batch
                x_batch = x_batch.T
                y_batch = y_batch.T

                # logit calc and apply ReLU
                z1 = np.dot(w1_mat, x_batch) + b1_vec
                a1 = self.relu_function(z1, self.hidden_units, self.minibatch_size)

                # logit calc and apply Softmax
                z2 = np.dot(w2_mat, a1) + b2_vec
                a2_softmax = self.softmax_function(z2)

                # cross-entropy for loss function
                gt_vector = np.zeros((10, self.minibatch_size))
                for example_num in range(self.minibatch_size):
                    gt_vector[y_batch[0, example_num], example_num] = 1

                # regularization for weights
                d_w2_mat = self.regularization_rate*w2_mat
                d_w1_mat = self.regularization_rate*w1_mat

                # backpropagation
                delta_2 = np.array(a2_softmax - gt_vector)
                d_w2_mat = d_w2_mat + np.dot(delta_2, (np.matrix(a1)).T)
                d_b2_vec = np.sum(delta_2, axis=1, keepdims=True)

                delta_1 = np.array(np.multiply((np.dot(w2_mat.T, delta_2)), self.grad_relu(z1, self.hidden_units, self.minibatch_size)))
                d_w1_mat = d_w1_mat + np.dot(delta_1, np.matrix(x_batch).T)
                d_b1_vec = np.sum(delta_1, axis=1, keepdims=True)

                d_w2_mat = d_w2_mat/self.minibatch_size
                d_w1_mat = d_w1_mat/self.minibatch_size
                d_b2_vec = d_b2_vec/self.minibatch_size
                d_b1_vec = d_b1_vec/self.minibatch_size

                # update weights
                w2_mat = w2_mat - self.learning_rate*d_w2_mat
                b2_vec = b2_vec - self.learning_rate*d_b2_vec

                w1_mat = w1_mat - self.learning_rate*d_w1_mat
                b1_vec = b1_vec - self.learning_rate*d_b1_vec

        self.w1_mat, self.b1_vec, self.w2_mat, self.b2_vec = w1_mat, b1_vec, w2_mat, b2_vec

    # function to test the MLP model
    def test(self, testX):
        output_labels = np.zeros(testX.shape[0])

        num_examples = testX.shape[0]

        testX = np.reshape(testX, (num_examples, 28*28))
        testX = testX.T

        # test with trained model
        z1 = np.dot(self.w1_mat, testX) + self.b1_vec    
        a1 = self.relu_function(z1, self.hidden_units, num_examples)

        z2 = np.dot(self.w2_mat, a1) + self.b2_vec
        a2_softmax = self.softmax_function(z2)

        for i in range(num_examples):
            pred_col = a2_softmax[:, [i]]
            output_labels[i] = np.argmax(pred_col)

        return output_labels
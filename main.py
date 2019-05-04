import numpy as np
import os
import model

def load_mnist():  # train 60k / test 10k
    data_dir = './'

    fd = open(os.path.join(data_dir, 'train-images-idx3-ubyte'))
    loaded = np.fromfile(file=fd, dtype=np.uint8)
    trX = loaded[16:].reshape((60000, 28, 28, 1)).astype(np.float)

    fd = open(os.path.join(data_dir, 'train-labels-idx1-ubyte'))
    loaded = np.fromfile(file=fd, dtype=np.uint8)
    trY = loaded[8:].reshape((60000)).astype(np.int)

    fd = open(os.path.join(data_dir, 't10k-images-idx3-ubyte'))
    loaded = np.fromfile(file=fd, dtype=np.uint8)
    teX = loaded[16:].reshape((10000, 28, 28, 1)).astype(np.float)

    fd = open(os.path.join(data_dir, 't10k-labels-idx1-ubyte'))
    loaded = np.fromfile(file=fd, dtype=np.uint8)
    teY = loaded[8:].reshape((10000)).astype(np.int)

    trY = np.asarray(trY)
    teY = np.asarray(teY)

    perm = np.random.permutation(trY.shape[0])
    trX = trX[perm]
    trY = trY[perm]

    perm = np.random.permutation(teY.shape[0])
    teX = teX[perm]
    teY = teY[perm]

    return trX, trY, teX, teY  # TrainX, TrainY, TestX, TestY

def main():
    trainX, trainY, testX, testY = load_mnist()
    print("Shapes: ", trainX.shape, trainY.shape, testX.shape, testY.shape)
    
    epochs = 25
    num_hidden_units = 300 
    minibatch_size = 100  
    regularization_rate = 0.01 
    learning_rate = 0.001 

    model = model.MLP(num_hidden_units, minibatch_size, regularization_rate, learning_rate)

    print("Starting training..........")
    model.train(trainX, trainY, epochs)
    print("Training complete")

    print("Starting testing..........")
    labels = model.test(testX)
    accuracy = np.mean((labels == testY)) * 100.0
    print("\nTest accuracy: %lf%%" % accuracy)

if __name__ == '__main__':
    main()
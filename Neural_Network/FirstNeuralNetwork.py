import numpy as np


# Ref: https://towardsdatascience.com/how-to-build-your-own-neural-network-from-scratch-in-python-68998a08e4f6
# Only Feed Forward

class FirstNeuralNetwork():
    '''
    Nural network
    input vector will be 1 * 3
    weights1 will be 3 * 4
    weights2 vector will be 4*1
    '''

    def __init__(self, x, y):
        self.input = x
        self.weights1 = np.random.rand(self.input.shape[1], 4)
        self.weights2 = np.random.rand(4, 1)

    ''' Implement sigmoid Function :
    Read every element of Matrix after liner combination of Input vector and weights vector 
    Implement Sigmoid function on Independent element and return a output array 
    '''

    def sigmoid_fn(self, i):

        i_arr = np.array(i)

        for j in range(np.shape(i_arr)[1]):
            sigmoid_val = 1 / (1 + np.exp(-(i_arr[0, j])))
            if sigmoid_val > 0.5:
                i_arr[0, j] = 0.1;
            else:
                i_arr[0, j] = 0;
        return i_arr;

    ''' Implement FeedForward function
    apply linear combination of Input and weight vector , apply sigmoid function and retur the the final output
    for all the layer of Neural network
    '''

    def FeedForward(self):

        self.layer1 = self.sigmoid_fn(np.dot(self.input, self.weights1))
        self.output = self.sigmoid_fn(np.dot(self.layer1, self.weights2))

def build_main():
    x = np.array([[0.7, 0.5, 0.2]])
    y = 1

    NN = FirstNeuralNetwork(x, y)

    print("Neural network Input Vector is {} and size is {}".format(NN.input, x.shape))
    print("Neural network layer 1 weight Vector is {}".format(NN.weights1))
    print("Neural network layer 2 Weight Vector is {}".format(NN.weights2))

    NN.FeedForward()
    print("Neural network layer 1 output Vector is {}".format(NN.layer1))
    print("Neural network layer 1 output Vector is {}".format(NN.output))

if __name__ == "__main__":
    build_main()


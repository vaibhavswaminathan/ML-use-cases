import numpy as np

# compute sigmoid nonlinearity
def sigmoid(x):
    output = 1 / (1 + np.exp(-x))
    return output

# convert output of sigmoid function to its derivative
def sigmoid_to_deriv(output):
    return output * (1 - output)

def predict(inp, weights):
    print(inp, sigmoid(np.dot(inp, weights)))

# input vectors with weights
X = np.array([ [0,1,1],
                [0,1,1],
                [1,0,0],
                [1,0,0]])
# output (target) vector
Y = np.array([[0,0,1,1]]).T

np.random.seed(1)

# init weights randomly with mean 0
weights0 = 2 * np.random.rand(3,1) - 1

for i in range(100):
    # forward propagation
    layer0 = X
    layer1 = sigmoid(np.dot(layer0, weights0))
    # compute the error
    layer1_error = layer1 - Y
    mse = 1/len(Y) * sum(np.square(layer1_error))
    print("Iteration %d ---> MSE Loss: %f" % (i,mse[0]))

    # gradient descent
    # calculate the slope at current x position
    layer1_delta = layer1_error * sigmoid_to_deriv(layer1) # this is the derivative of the MSE loss function
    weights0_deriv = np.dot(layer0.T, layer1_delta)
    # change x by the negative of the slope (x = x - slope)
    weights0 -= weights0_deriv

print('INPUT   PREDICTION')
predict([0,1,1], weights0)
predict([1,0,0], weights0)
#test prediction of the unknown data
predict([1,1,0], weights0)
predict([0,0,1], weights0)
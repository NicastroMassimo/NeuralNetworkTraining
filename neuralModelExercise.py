from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Input
from keras.utils import to_categorical
from keras.optimizers import SGD

# load the MNIST dataset which is luckily stored in keras
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Non capisco ancora questo passo
X_train = X_train.reshape((X_train.shape[0], 28 * 28))
X_test = X_test.reshape((X_test.shape[0], 28 * 28))

# convertiamo da indici a float perchè è il formato supportato in keras
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

# normalize to range 0-1, so as to better compute the gradient
X_train = X_train / 255.0
X_test = X_test / 255.0

# one hot encode target values, since we are going to use the softmax, as it is as 10 class classification 
# problem, (1 for each digit)
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# define the model
N = 32 # number of neurons in the hidden layer
model = Sequential()

# the shape of the input is defined, so that the weights can be defined
model.add(Input(shape=(28 * 28,)))
model.add(Dense(N, activation='relu'))
model.add(Dense(10, activation='softmax')) #we use softmax as said before for that reasons

# compile the model
sgd = SGD(learning_rate=0.01) # Define how fast the stochastic gradient descents, converges
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy']) # <-- still to understand

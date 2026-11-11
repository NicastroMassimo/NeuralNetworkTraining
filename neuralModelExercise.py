from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Input
from keras.utils import to_categorical
from keras.optimizers import SGD
import matplotlib.pyplot as plt #<-- Needed for visualization

# load the MNIST dataset which is luckily stored in keras
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Flatten the input to a 784 long array
X_train = X_train.reshape((X_train.shape[0], 28 * 28))
X_test = X_test.reshape((X_test.shape[0], 28 * 28))


# Convert from integer to float as it is the format better supported by keras
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
N = 16 # number of neurons in the hidden layer
model = Sequential()

# the shape of the input is defined, so that the input neurons can be defined
model.add(Input(shape=(28 * 28,)))
model.add(Dense(N, activation='relu'))
model.add(Dense(len(y_train[0]), activation='softmax')) #we use softmax as said before for that reasons

# compile the model
sgd = SGD(learning_rate=0.01) # Define how fast the stochastic gradient descents, converges
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy']) # <-- still to understand

# fit the model, define the epoch, and the number of sample taken before it's time to update the weight
history = model.fit(X_train, y_train, epochs=128, batch_size=128)

# evaluate the model
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f'Test Accuracy: {accuracy * 100:.2f}%')


# plot the accuracy and the loss curves
fig, ax1 = plt.subplots()
ax2 = ax1.twinx()
ax1.plot(history.history['accuracy'], 'g-')
ax2.plot(history.history['loss'], 'b-')
ax1.set_xlabel('Epochs')
ax1.set_ylabel('Accuracy', color='g')
ax2.set_ylabel('Loss', color='b')
plt.title('Model Accuracy and Loss')
plt.show()
plt.savefig('model_accuracy_loss.png') # To save the image in case the OS doesn't support a visualizer


#Importing important modules
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout,Flatten
from keras.layers import Conv2D,MaxPooling2D
from keras.callbacks import ModelCheckpoint,EarlyStopping
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense

#imstalling Tensorboard for collab
from google.colab import drive
drive.mount('/content/drive')

# the data, split betweentrain and test sets
(x_train,y_train),(x_test,y_test)=mnist.load_data()

#visualisation
%matplotlib inline

import matplotlib.pyplot as plt
w=10
h=10
fig=plt.figure(figsize=(8,8))
columns=10
rows=10
for i in range(1,columns*rows +1):
  img=x_test[i]
  fig.add_subplot(rows,columns,i)
  plt.imshow(img,cmap='gray')

plt.show()



#input image dimensions
img_rows,img_cols=28,28

#Keras espects data to be in the format (N_E.N_H,N_W,N_C)
#N_E=Number of Examples, N_H=height,N_W=width,N_C=Number of channels
x_train=x_train.reshape(x_train.shape[0],img_rows,img_cols,1)
x_test=x_test.reshape(x_test.shape[0],img_rows,img_cols,1)
input_shape=(img_rows,img_cols,1)

#normalisation
x_train=x_train.astype('float32')
x_test=x_test.astype('float32')

#normalisation input
x_train /= 255.0
x_test /= 255.0
print('x_train shape:',x_train.shape)
print(x_train.shape[0],'train samples')
print(x_test.shape[0],'test samples')

batch_size=128
num_classes=10
epochs=12

print(y_train[0])

#Build The CNN
#Define layers

# Define input shape (example for 28x28 grayscale images)
input_shape = (28, 28, 1)
num_classes = 10  # Example for 10-class classification

# Initialize the model
model = Sequential()

# Add a convolutional layer with 32 filters of size 3x3 and ReLU activation
model.add(Conv2D(32, kernel_size=(3,3), activation='relu', input_shape=input_shape))

# Add a convolutional layer with 64 filters of size 3x3 and ReLU activation
model.add(Conv2D(64, (3,3), activation='relu'))

# Add a max pooling layer of size 2x2
model.add(MaxPooling2D(pool_size=(2,2)))

# Apply dropout with 0.25 probability
model.add(Dropout(0.25))

# Flatten the layer
model.add(Flatten())

# Add a fully connected layer with 128 units and ReLU activation
model.add(Dense(128, activation='relu'))

# Apply dropout with 0.5 probability
model.add(Dropout(0.5))

# Add fully connected output layer with 10 units and softmax activation
model.add(Dense(num_classes, activation='softmax'))

# Print the model summary
model.summary()

# Import necessary libraries
import tensorflow as tf
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.optimizers import Adam  # Or use Adamax if needed

# Define optimizer (Adam with learning rate 0.001)
opt = Adam(learning_rate=0.001)

# Compile the model with the correct optimizer and loss function
model.compile(loss=categorical_crossentropy,
              optimizer=opt,  # Fixed: Use the correct variable name
              metrics=['accuracy'])

# Print model summary to confirm
model.summary()


from keras.datasets import mnist
import numpy as np
import matplotlib.pyplot as plt
from conv3x3 import Conv3x3
from maxpool import MaxPool2D
from softmax import Softmax
from tqdm import tqdm

(X_train, Y_train), (X_test, Y_test) = mnist.load_data()

X_train_now = X_train[0:2000]
Y_train_now = Y_train[0:2000]
X_test_now = X_test[0:2000]
Y_test_now = Y_test[0:2000]

conv3x3 = Conv3x3(16)
maxpool2d = MaxPool2D()
softmax = Softmax(13*13*16, 10)

# single feed forward CNN impleamnetation

def single_full_forward(image, label):
    image = image/255
    
    out_image = conv3x3.single_conv_forward(image)
    out_image = maxpool2d.maxpool_forward(out_image)
    probabilities = softmax.softmax_forward(out_image)

    loss = -np.log(probabilities[label])
    acc = 1 if np.argmax(probabilities) == label else 0

    return probabilities, loss, acc

# single training impleamentation

def train(image, label, learning_rate = 0.003):
    probabilities, loss, acc = single_full_forward(image, label)
    gradients = np.zeros(10)
    gradients[label] = -1/probabilities[label]

    # backprop
    gradient_dx = softmax.soft_backward(gradients, learning_rate)
    gradient_dx = maxpool2d.maxpool_backward(gradient_dx)
    gradient_dx = conv3x3.single_conv_backward(gradient_dx, learning_rate)
    return loss, acc

# main function

print(".... USING KERAS DATASET ONLY, NO OTHER FUNCTIONALITIES ARE BEING ADDED HERE ....")
print("\nCONVOLUTON INITIALIZED ...")
print("TRAINING FOR ", len(X_train_now), " IMAGES \n")
print("USING ONE CONV LAYER WITH 64 FILTERS")
print("NO DENSE LAYER\n")

for epoch in range(3):
    permutation = np.random.permutation(len(X_train_now))
    train_images = X_train_now[permutation]
    train_labels = Y_train_now[permutation]
    val_images = X_test_now[permutation]
    val_labels = Y_test_now[permutation]

    # TRAIN
    total_train_loss = 0
    train_accuracy = 0
    total_val_loss = 0
    val_accuracy = 0

    for i, (image_train, label_train) in enumerate(zip(train_images, train_labels)):
        if i> 0 and i%100 == 99:
            print(
                "AT EPOCH ", epoch, " AFTER COMPUTING ", i+1, " IMAGES TRAIN LOSS IS ", 
                total_train_loss/100, " TRAIN ACC ", train_accuracy, "%"
            )
            total_train_loss = 0
            train_accuracy = 0

        loss_train, acc_train = train(image_train, label_train)

        total_train_loss += loss_train
        train_accuracy += acc_train
    print("\n")


val_loss = 0
val_accuracy = 0
for i, (image, label) in enumerate(zip(X_test_now, Y_test_now)):
    _, loss, acc = single_full_forward(image, label)
    val_loss += loss
    val_accuracy += acc

num_test = len(X_test_now)
print("TOTAL TEST LOSS: ", val_loss/num_test)
print("TOTAL TEST ACCURACY: ", (val_accuracy/num_test) *100 , "%")
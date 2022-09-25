from __future__ import print_function
import glob
import cv2
import os
from keras.callbacks import LearningRateScheduler
from gc import callbacks
import tensorflow as tf
from tensorflow.python.data import Iterator
from datetime import datetime
from matplotlib import image
from torchvision import datasets, transforms
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import torch.nn as nn
import torch
from tensorflow.keras import datasets, layers, models
from tensorflow.keras.datasets import cifar10
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Dense, Flatten, Activation, BatchNormalization
import numpy as np
import random
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from google.colab import drive
from google.colab import auth
import cv2 as cv

# Loss by which we use to create FGSM examples
loss_object = tf.keras.losses.CategoricalCrossentropy()


def create_adversarial_pattern(model, input_image, input_label):
    with tf.GradientTape() as tape:
        tape.watch(input_image)

        # Model's accuracy on image and subsequent loss value
        prediction = model(input_image)
        loss = loss_object(input_label, prediction)

    # Get the gradients of the loss with respect to the input image.
    gradient = tape.gradient(loss, input_image)
    # Get the sign of the gradients to create the perturbation
    signed_grad = tf.sign(gradient)
    return signed_grad


def generate_adversarials(model, batch_size):
    x = []
    y = []
    while True:
        for batch in range(batch_size):
            # Take the label and then mislabel it with another random label
            label = y_train[batch]
            N = random.randint(1, 10)
            label = (N + label) % 10

            ####################################################
            # Epsilon is the strength of the FGSM attack       #
            # epsilons = [0.01, 0.1, 0.15]                     #
            ####################################################

            # Image to be adversarialy perturbed
            image = x_train[N]
            # Strength of Attack
            eps = 0.01
            # Adversarial Pattern Generation
            perturbations = create_adversarial_pattern(model, x_train[N].reshape(
                (1, img_rows, img_cols, channels)), label).numpy()
            # Application to base image to create adversarial image
            adv_x = image + (eps*perturbations)
            adv_x = tf.clip_by_value(adv_x, -1, 1)
            # Make a list of Adversarial images
            x.append(adv_x)
            y.append(y_train[N])

            # Make numpy arrays and return
            x = np.asarray(x).reshape(
                (batch_size, img_rows, img_cols, channels))
            y = np.asarray(y)
            yield x, y


"""## Standard Model for Reference"""

class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

plt.figure(figsize=(10, 10))
for i in range(25):
    plt.subplot(5, 5, i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i])
    # The CIFAR labels happen to be arrays,
    # which is why you need the extra index
    plt.xlabel(class_names[train_labels[i][0]])
plt.show()

model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10))

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(
                  from_logits=True),
              metrics=['accuracy'])


def lr_schedule(epoch):
    lrate = 0.001
    if epoch > 75:
        lrate = 0.0005
    if epoch > 100:
        lrate = 0.0003
    return lrate


history = model.fit(train_images, train_labels, epochs=125,
                    validation_data=(test_images, test_labels), callbacks=[LearningRateScheduler(lr_schedule)])

plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.5, 1])
plt.legend(loc='lower right')

test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)

# accuracy was 71%

"""# **Pytorch**"""

drive.mount('/content/drive')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# CONSTANTS

lr = 0.001  # values between 0.001 and 0.00001
wdecay = 0.005  # L2 regularization

batchSize = 32  # depends on the GPU memory
epochs = 1000  # 0 for testing
valEpochs = 5  # number of epochs until validation (0 for no validation)

imSize = 128  # IMAGE_SIZE x IMAGE_SIZE, None = Different sizes
channel = 3  # 3 = RGB, 1 = grayscale

num_classes = 2

saveEpoch = 10  # number of epochs until saving a checkpoint
printEpoch = 5  # number of epochs until printing accuracy

eps = [0.01, 0.1, 0.15]

images = []
labels = []

dire = "/content/drive/MyDrive/Colab_Files/Adversarial/cars_small"

for filename in os.listdir(dire)[:5]:
    if filename.endswith("png"):
        # car Images
        im = image.imread(dire + "/" + filename)
        images.append(im)
        labels.append(1)

dire = "/content/drive/MyDrive/Colab_Files/Adversarial/peds_small"

for filename in os.listdir(dire)[:5]:
    if filename.endswith("png"):
        # Pedestrian Images
        im = image.imread(dire + "/" + filename)
        images.append(im)
        labels.append(0)

train_images, train_labels, test_images, test_labels = train_test_split(
    images, labels, test_size=0.2, random_state=4)

train_images = torch.tensor(train_images)
test_images = torch.tensor(test_images)
train_labels = torch.tensor(train_labels)
test_labels = torch.tensor(test_labels)

# FGSM attack code


def fgsm_attack(image, epsilon, data_grad):
    # Collect the element-wise sign of the data gradient
    sign_data_grad = data_grad.sign()
    # Create the perturbed image by adjusting each pixel of the input image
    perturbed_image = image + epsilon*sign_data_grad
    # Adding clipping to maintain [0,1] range
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    # Return the perturbed image
    return perturbed_image


class simplenet(nn.Module):
    def __init__(self, classes=10, simpnet_name='simplenet'):
        super(simplenet, self).__init__()
        # print(simpnet_name)
        # self._make_layers(cfg[simpnet_name])
        self.features = self._make_layers()
        self.classifier = nn.Linear(256, classes)
        self.drp = nn.Dropout(0.1)

    def load_my_state_dict(self, state_dict):

        own_state = self.state_dict()

        # print(own_state.keys())
        # for name, val in own_state:
        # print(name)
        for name, param in state_dict.items():
            name = name.replace('module.', '')
            if name not in own_state:
                # print(name)
                continue
            if isinstance(param, Parameter):
                # backwards compatibility for serialized parameters
                param = param.data
            print("STATE_DICT: {}".format(name))
            try:
                own_state[name].copy_(param)
            except:
                print('While copying the parameter named {}, whose dimensions in the model are'
                      ' {} and whose dimensions in the checkpoint are {}, ... Using Initial Params'.format(
                          name, own_state[name].size(), param.size()))

    def forward(self, x):
        out = self.features(x)

        # Global Max Pooling
        out = F.max_pool2d(out, kernel_size=out.size()[2:])
        # out = F.dropout2d(out, 0.1, training=True)
        out = self.drp(out)

        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def _make_layers(self):

        self.model = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=[3, 3],
                      stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(64, eps=1e-05, momentum=0.05, affine=True),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 128, kernel_size=[3, 3],
                      stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(128, eps=1e-05, momentum=0.05, affine=True),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, 128, kernel_size=[
                      3, 3], stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(128, eps=1e-05, momentum=0.05, affine=True),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, 128, kernel_size=[
                      3, 3], stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(128, eps=1e-05, momentum=0.05, affine=True),
            nn.ReLU(inplace=True),


            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2),
                         dilation=(1, 1), ceil_mode=False),
            nn.Dropout2d(p=0.1),


            nn.Conv2d(128, 128, kernel_size=[
                      3, 3], stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(128, eps=1e-05, momentum=0.05, affine=True),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, 128, kernel_size=[
                      3, 3], stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(128, eps=1e-05, momentum=0.05, affine=True),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, 256, kernel_size=[
                      3, 3], stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(256, eps=1e-05, momentum=0.05, affine=True),
            nn.ReLU(inplace=True),



            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2),
                         dilation=(1, 1), ceil_mode=False),
            nn.Dropout2d(p=0.1),


            nn.Conv2d(256, 256, kernel_size=[
                      3, 3], stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(256, eps=1e-05, momentum=0.05, affine=True),
            nn.ReLU(inplace=True),


            nn.Conv2d(256, 256, kernel_size=[
                      3, 3], stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(256, eps=1e-05, momentum=0.05, affine=True),
            nn.ReLU(inplace=True),



            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2),
                         dilation=(1, 1), ceil_mode=False),
            nn.Dropout2d(p=0.1),



            nn.Conv2d(256, 512, kernel_size=[
                      3, 3], stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(512, eps=1e-05, momentum=0.05, affine=True),
            nn.ReLU(inplace=True),



            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2),
                         dilation=(1, 1), ceil_mode=False),
            nn.Dropout2d(p=0.1),


            nn.Conv2d(512, 2048, kernel_size=[
                      1, 1], stride=(1, 1), padding=(0, 0)),
            nn.BatchNorm2d(2048, eps=1e-05, momentum=0.05, affine=True),
            nn.ReLU(inplace=True),



            nn.Conv2d(2048, 256, kernel_size=[
                      1, 1], stride=(1, 1), padding=(0, 0)),
            nn.BatchNorm2d(256, eps=1e-05, momentum=0.05, affine=True),
            nn.ReLU(inplace=True),


            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2),
                         dilation=(1, 1), ceil_mode=False),
            nn.Dropout2d(p=0.1),


            nn.Conv2d(256, 256, kernel_size=[
                      3, 3], stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(256, eps=1e-05, momentum=0.05, affine=True),
            nn.ReLU(inplace=True),

        )

        for m in model.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(
                    m.weight.data, gain=nn.init.calculate_gain('relu'))

        return model

    # Function to save the model
    def saveModel():
        path = "./myFirstModel.pth"
        torch.save(model.state_dict(), path)

    # Function to test the model with the test dataset and print the accuracy for the test images
    def testAccuracy():

        model.eval()
        accuracy = 0.0
        total = 0.0

        with torch.no_grad():
            for data in len(train_images):
                images, labels = data
                # run the model on the test set to predict labels
                outputs = model(images)
                # the label with the highest energy will be our prediction
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                accuracy += (predicted == labels).sum().item()

        # compute the accuracy over all test images
        accuracy = (100 * accuracy / total)
        return(accuracy)

    # Training function. We simply have to loop over our data iterator and feed the inputs to the network and optimize.

    def train(num_epochs):

        best_accuracy = 0.0

        # Define your execution device
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print("The model will be running on", device, "device")
        # Convert model parameters and buffers to CPU or Cuda
        model.to(device)

        for epoch in range(num_epochs):  # loop over the dataset multiple times
            running_loss = 0.0
            running_acc = 0.0

            for i, (images, labels) in enumerate(train_loader, 0):

                # get the inputs
                images = Variable(images.to(device))
                labels = Variable(labels.to(device))

                # zero the parameter gradients
                optimizer.zero_grad()
                # predict classes using images from the training set
                outputs = model(images)
                # compute the loss based on model output and real labels
                loss = loss_fn(outputs, labels)
                # backpropagate the loss
                loss.backward()
                # adjust parameters based on the calculated gradients
                optimizer.step()

                # Let's print statistics for every 1,000 images
                running_loss += loss.item()     # extract the loss value
                if i % 1000 == 999:
                    # print every 1000 (twice per epoch)
                    print('[%d, %5d] loss: %.3f' %
                          (epoch + 1, i + 1, running_loss / 1000))
                    # zero the loss
                    running_loss = 0.0

            # Compute and print the average accuracy fo this epoch when tested over all 10000 test images
            accuracy = testAccuracy()
            print('For epoch', epoch+1,
                  'the test accuracy over the whole test set is %d %%' % (accuracy))

            # we want to save the model if the accuracy is the best
            if accuracy > best_accuracy:
                saveModel()
                best_accuracy = accuracy


def test(model, device, test_loader, epsilon):

    # Accuracy counter
    correct = 0
    adv_examples = []

    # Loop over all examples in test set
    for data, target in test_loader:

        # Send the data and label to the device
        data, target = data.to(device), target.to(device)

        # Set requires_grad attribute of tensor. Important for Attack
        data.requires_grad = True

        # Forward pass the data through the model
        output = model(data)
        # get the index of the max log-probability
        init_pred = output.max(1, keepdim=True)[1]

        # If the initial prediction is wrong, dont bother attacking, just move on
        if init_pred.item() != target.item():
            continue

        # Calculate the loss
        loss = F.nll_loss(output, target)

        # Zero all existing gradients
        model.zero_grad()

        # Calculate gradients of model in backward pass
        loss.backward()

        # Collect datagrad
        data_grad = data.grad.data

        # Call FGSM Attack
        perturbed_data = fgsm_attack(data, epsilon, data_grad)

        # Re-classify the perturbed image
        output = model(perturbed_data)

        # Check for success
        # get the index of the max log-probability
        final_pred = output.max(1, keepdim=True)[1]
        if final_pred.item() == target.item():
            correct += 1
            # Special case for saving 0 epsilon examples
            if (epsilon == 0) and (len(adv_examples) < 5):
                adv_ex = perturbed_data.squeeze().detach().cpu().numpy()
                adv_examples.append(
                    (init_pred.item(), final_pred.item(), adv_ex))
        else:
            # Save some adv examples for visualization later
            if len(adv_examples) < 5:
                adv_ex = perturbed_data.squeeze().detach().cpu().numpy()
                adv_examples.append(
                    (init_pred.item(), final_pred.item(), adv_ex))

    # Calculate final accuracy for this epsilon
    final_acc = correct/float(len(test_loader))
    print("Epsilon: {}\tTest Accuracy = {} / {} = {}".format(epsilon,
          correct, len(test_loader), final_acc))

    # Return the accuracy and an adversarial example
    return final_acc, adv_examples


model = simplenet()
model.fit(10, train_images, train_labels, test_images, test_labels)


a

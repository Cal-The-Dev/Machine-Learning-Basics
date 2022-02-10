# -*- coding: utf-8 -*-
"""
Created on Fri Nov 16 00:50:15 2018

@author: Callum Smith - Student ID 200969669
"""
#Import everything you can possibly think of

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import os
import skimage
import numpy
import scipy
import sklearn
from sklearn import metrics
import matplotlib
import pandas
import random
import pickle
# Import as np and plt for numpy and matplotlib and get the transform from skimage
import numpy as np
import matplotlib.pyplot as plt 
from skimage import transform 
from skimage.color import rgb2gray

from sklearn.neighbors import KNeighborsClassifier



#This function is my loading function, it opens the cfar batch file, using a with statement
# It uses pickle to open the file (As explained to me on cfars website, as that is how they store their dataset)
# and once it's loaded as it's done with a with statement, it cleans up after itself.
def load_cfar10_batch(cifar10_dataset_folder_path, batch_id):
    with open(cifar10_dataset_folder_path + '/data_batch_' + str(batch_id), mode='rb') as file:
        # The encoding for the cfar file, which is stored as a pickle file, is in latin1
        #Using pickle I load the batch file
        batch = pickle.load(file, encoding='latin1')
        #Reshapre the data to be the correct size for our tensor
    features = batch['data'].reshape((len(batch['data']), 3, 32, 32)).transpose(0, 2, 3, 1)
    #Grab the labels
    labelsn = batch['labels']
        
    return features, labelsn
# Now it runs the function to return the features and labelsn to "images" and "labels" respectively
print("Loading 30000 images from the Cfar10 Dataset. Please stand by.")
images, labels = load_cfar10_batch("cifar-10-python", 1)
images2, labels2 = load_cfar10_batch("cifar-10-python", 2)
images3, labels3 = load_cfar10_batch("cifar-10-python", 3)




#Similar the above one, only it loads up the test data stored in the test_batch file instead of the standard data
def load_cfar10_test(cifar10_dataset_folder_path):
    with open(cifar10_dataset_folder_path + '/test_batch', mode='rb') as file:
        # The encoding for the cfar file, which is stored as a pickle file, is in latin1
        #Using pickle I load the batch file
        batch = pickle.load(file, encoding='latin1')
        #Reshape it all
    features = batch['data'].reshape((len(batch['data']), 3, 32, 32)).transpose(0, 2, 3, 1)
    #Grab the labels
    labelsn = batch['labels']
    #Return the features and labels
        
    return features, labelsn
#Set test_images and test_labels to the loaded images and labels from the load_cfar10_test function
test_images, test_labels = load_cfar10_test("cifar-10-python")



# Grab the images, rescale them as 32x32 (Not strictly necassary as these images are presized already)
# But useful for normalizing the data and for use with any other data sets.
images32 = [transform.resize(image, (32, 32)) for image in images]
images32two = [transform.resize(image, (32, 32)) for image in images2]
images32three = [transform.resize(image, (32, 32)) for image in images3]

testimages32 = [transform.resize(image, (32, 32)) for image in test_images]

# Convert `images32` to a numpy array to allow it to print the actual data instead of the placeholder
images32 = np.array(images32)
images32two = np.array(images32two)
images32three = np.array(images32three)
testimages32 = np.array(testimages32)

# Convert `images28` to grayscale - Use this for testing accuracy between colour/none
# My accuracy goes down for cfar in grayscale, as such I am commenting this out.
#images32 = rgb2gray(images32)
#testimages32 = rgb2gray(testimages32)


#This function is just to load the label names for the example labels as well as change the numeric prediction and truth's to
# their label equivalents.
def load_label_names():
    return ['Plane', 'Car', 'Bird', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck']

# Get the first of each label category by formatting it as a set
first_of_each_label = set(labels)

# Initialize the figure
plt.figure(figsize=(15, 15))

#Set a counter

i = 1
# For the first of each label, print it out and classify what it is
for label in first_of_each_label:
    # You pick the first image for each label
    image = images32[labels.index(label)]
    # Define 64 subplots 
    plt.subplot(8, 8, i)
    # Remove the Axis
    plt.axis('off')
    # Add the titles. 
    plt.title("Label {0} ({1})".format(label, load_label_names()[label]))
    # Add 1 to the counter
    i += 1
    # And you plot this first image - Note cmap = "gray" here only applies if we're grayscaling
    # Which in this example we are not, however I plan to use this code in assignment 2 for that purpose
    # however its function remains inert until grayscaling is applied.
    plt.imshow(image, cmap = "gray")
    
# Show the plot
plt.show()

#Neural Network code here

# Initialize placeholders - Add ,3 to x if you use colour and remove it if you do not, as that's a more advanced tensor
x = tf.placeholder(dtype = tf.float32, shape = [None, 32, 32, 3])
y = tf.placeholder(dtype = tf.int32, shape = [None])

# Flatten the input data in order to normalize it by multiplying its height, width and channels into 1 variable.
images_flat = tf.compat.v1.estimator.layers.flatten(x)

# Fully connected layer of logits (Probability mapping function that gives negative or positive based on the probability -
# - being less than or more than and equal to 0.5), it then normalizes this data into the correct type of data needed below
# For the argmax function
logits = tf.compat.v1.estimator.layers.fully_connected(images_flat, 62, tf.nn.relu)

# Define a loss function, it measures the distance between the model outputs and the target (truth) values.
loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels = y, 
                                                                    logits = logits))
# Use Adam Optimizer, with a learning rate (I changed from 0.0001 to 0.005 as it seemed to improve accuracy slightly)
# With the training focused on having the smallest loss function, aka, comparing output with the target value and trying
# to find the smallest amount of loss (Smallest amount of difference in the image) to make a prediction
train_op = tf.train.AdamOptimizer(learning_rate=0.005).minimize(loss)

# Convert logits to label indexes using the argmax function to return the largest value in the first dimension of logits
correct_pred = tf.argmax(logits, 1)

# Define what our accuracy is.
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

#Set a random seed to make sure that our results stay consistent between runs.
tf.set_random_seed(1337)
#Set up a session and run the global variables initializer to allow us to initalize other variables down in the op.
sess = tf.Session()
sess.run(tf.global_variables_initializer())
#Train the model 300 times, print progress as we go.
for i in range(200):
        _, accuracy_val = sess.run([train_op, accuracy], feed_dict={x: images32, y: labels})
        if i % 10 == 0:
            print("Working . . . {} out of 200 Runs complete for batch 1.".format(i))

for i in range(200):
        _, accuracy_val = sess.run([train_op, accuracy], feed_dict={x: images32two, y: labels2})
        if i % 10 == 0:
            print("Working . . . {} out of 200 Runs complete for batch 2.".format(i))
            
for i in range(200):
        _, accuracy_val = sess.run([train_op, accuracy], feed_dict={x: images32three, y: labels3})
        if i % 10 == 0:
            print("Working . . . {} out of 200 Runs complete for batch 3.".format(i))

print(" Printing 10 sample predictions from test image set against the model trained on 3 batches of 10k images")
# Pick 10 random images
sample_indexes = random.sample(range(len(testimages32)), 10)
sample_images = [testimages32[i] for i in sample_indexes]
sample_labels = [test_labels[i] for i in sample_indexes]

# Run the "correct_pred" operation from before.
#predicted = sess.run([correct_pred], feed_dict={x: sample_images})[0] 
                   
# Print the real and predicted labels - Originally had this and keeping it in for debugging purposes, but, commenting it out
# due to it not being needed with the visual predictions below it.
#print(sample_labels)
#print(predicted)

# Run predictions against the full test set.
predictedtest = sess.run([correct_pred], feed_dict={x: testimages32})[0]

# Display the predictions and the ground truth visually.
fig = plt.figure(figsize=(15, 15))

for i in range(len(sample_images)):
    #Set the actual label and predicted labels
    truth = test_labels[i]
    prediction = predictedtest[i]
    #Plot the graph and turn the Axis off
    plt.subplot(6, 3,1+i)
    
    plt.axis('off')
    color='red' if truth != prediction else 'green'
    plt.text(40, 10, "Truth:        {0}\nPrediction: {1}".format(load_label_names()[truth], load_label_names()[prediction]), 
             fontsize=12, color=color)
    
    #Once again we have the gray colourmap in case we need it in future, but it will be inert for now as we commented out the
    #grayscaling function as it's not required in assignment 1 for this dataset
    plt.imshow(test_images[i],  cmap="gray")
#Show the graph
plt.show()


# Calculate correct matches 
match_count = sum([int(y == y_) for y, y_ in zip(test_labels, predictedtest)])

# Calculate the accuracy
accuracy = match_count / len(test_labels)

# Print the accuracy
print(str(match_count) + " correct matches out of " + str(len(test_labels)) + " images tested.")
print("Accuracy: {:.3f}".format(accuracy))

#Print the confusion matrix
print("C Matrix:\n%s" % metrics.confusion_matrix(test_labels, predictedtest))



#KNN CLASSIFIER
print("K-NN Classifier")
KNN = KNeighborsClassifier()

print("Fitting data to model, please stand by")
KNN.fit(images32.reshape(10000, -1), labels)
KNN.fit(images32two.reshape(10000, -1), labels2)
KNN.fit(images32three.reshape(10000, -1), labels3)

print("Predicting classes for 10000 test data images, please stand by, this may take several minutes, like seriously, go for a coffee or something.")
print("Average time for this process is about 7-9 minutes")
KNNpredicted = KNN.predict(testimages32.reshape(10000, -1))

# Calculate correct matches 
match_count = sum([int(y == y_) for y, y_ in zip(test_labels, KNNpredicted)])

# Calculate the accuracy
accuracy = match_count / len(test_labels)

# Print the accuracy
print(str(match_count) + " correct matches out of " + str(len(test_labels)) + " images tested.")
print("Accuracy: {:.3f}".format(accuracy))

#Print the confusion matrix
print("C Matrix:\n%s" % metrics.confusion_matrix(test_labels, KNNpredicted))



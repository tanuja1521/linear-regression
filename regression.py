#importing necessary libraries
import numpy as np 
import tensorflow as tf 
import matplotlib.pyplot as plt

np.random.seed(101)
tf.set_random_seed(101)

#Generating and random linear data(50 data points ranging from 0 to 50)
#adding noise to random linear data

x = np.linspace(0,50.50)
x += np.random.uniform(-4,4,50)

#Generating value of y randomly from standard normal distribution

y = np.random.normal(0,1,50)

# Number of points
n = len(x)

#Plotting the training data
plt.scatter(x, y) 
plt.xlabel('x') 
plt.ylabel('y') 
plt.title("Training Data") 
plt.show()

#Creating placeholders for x and y
X = tf.placeholder("float") 
Y = tf.placeholder("float")

#Declaring variables weight and bias randomly
W = tf.Variable(np.random.randn(), name = "W") 
b = tf.Variable(np.random.randn(), name = "b")

#Definig learning rate and number of epochs

learning_rate = 0.01
training_epochs = 1000

#Hypothesis

y_pred = tf.add(tf.multiply(X,W), b)

#Mean Squared error Cost Function

cost = tf.reduce_sum(tf.pow(y_pred - Y , 2))/(2*n)

#Gradient Descent OptimiZer

optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

# Global Variables Initializer

init = tf.global_variables_initializer()

# Starting the Tensorflow Session

with tf.Session() as sess:
    # Initializing the Variables
    sess.run(init) 
    # Iterating through all the epochs
    for epoch in range(training_epochs): 
        # Feeding each data point the optimizer by using Feed Dictionary
        for (_x, _y) in zip(x, y): 
            sess.run(optimizer, feed_dict = {X : _x, Y : _y}) 
        # Displaying the result after every 50 epochs 
        if (epoch + 1) % 50 == 0: 
            # Calculating the cost a every epoch 
            c = sess.run(cost, feed_dict = {X : x, Y : y}) 
            print("Epoch", (epoch + 1), ": cost =", c, "W =", sess.run(W), "b =", sess.run(b)) 

    # Storing necessary values to be used outside the Session 
    training_cost = sess.run(cost, feed_dict ={X: x, Y: y}) 
    weight = sess.run(W) 
    bias = sess.run(b)

# Calculating the predictions 
predictions = weight * x + bias 
print("Training cost =", training_cost, "Weight =", weight, "bias =", bias, '\n') 

# Plotting the Results 
plt.plot(x, y, 'ro', label ='Original data') 
plt.plot(x, predictions, label ='Fitted line') 
plt.title('Linear Regression Result') 
plt.legend() 
plt.show() 

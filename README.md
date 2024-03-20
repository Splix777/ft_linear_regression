<h1 align="center">ft_linear_regression</h1>

<p align="center">
  <img src="images/linear_regression.png" alt="Your Image" width="300" height="200">
</p>

## An introduction to machine learning

### Summary
In this project, you will implement your first machine learning algorithm.

### Introduction
Machine learning is a growing field of computer science that may seem a bit complicated and reserved only to mathematicians. You may have heard of neural networks or k-means clustering and don’t understand how they work or how to code these kinds of algorithms... But don’t worry, we are actually going to start with a simple, basic machine learning algorithm.

### Objectives
The aim of this project is to introduce you to the basic concept behind machine learning. For this project, you will have to create a program that predicts the price of a car by using a [linear function](https://en.wikipedia.org/wiki/Linear_function) trained with a [gradient descent algorithm](https://en.wikipedia.org/wiki/Gradient_descent).

We will work on a precise example for the project, but once you’re done you will be able to use the algorithm with any other dataset.

### General Instructions
- You must create a program that predicts the price of a car by using a linear function.
- You are free to use any libraries you want as long as you can explain your choice and they don't do all the work for you. For example, the use of Python's `numpy.polyfit` is considered cheating.
- You should use a language that you are comfortable with; the whole project must be written in the same language.

### Mandatory part
- You will implement a simple linear regression with a single feature - in this case, the mileage of the car.
    - The first program will be used to predict the price of a car for a given mileage. When you launch the program, it should prompt you for a mileage, and then give you back the estimated price for that mileage. The program will use the following hypothesis to predict the price:
        `estimatePrice(mileage) = θ0 + (θ1 * mileage)`
        Before the run of the training program, theta0 and theta1 will be set to 0.
    - The second program will be used to train your model. It will read your dataset file and perform a linear regression on the data. Once the linear regression has completed, you will save the variables θ0 and θ1 for use in the first program.

### Bonus
Here are some bonuses that could be very useful:
- Plotting the data into a graph to see their distribution.
- Plotting the line resulting from your linear regression into the same graph to see the result of your hard work!
- A program that calculates the precision of your algorithm.

### Usage
```sh
make run

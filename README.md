# CORE Skills Data Science Springboard - Day 10

[![Binder](https://mybinder.org/badge.svg)](https://mybinder.org/v2/gh/core-skills/08-time-network-analysis.git/master)

Today is our hands-on a machine learning project! Hopefully you brought some interesting datasets you would like to work with. If not, we are providing a Petrophysical dataset (check the data folder for a description) and proposing some questions to be investigated with this data. 

We have several aims for today's session. We hope it will help you to have a clear picture of the main steps when developing a data science with machine learning project. More specifically, our aims for today are:

- Review data science and ML landscape and techniques
- Planning a ML data science project for your data
-  Learn to address data science questions:
	- How to frame the problem?  What are the questions I want to address with my data?
	- How to identify problems with the data: what are the data cleaning stages that I will have to do? 
	- How could I explore the data? How can I visualize my data to search for correlations?
	- How can I prepare my data for the ML algorithms?
	- What are meaningful evaluation metrics that I can apply?

- Once your data is prepared for the ML algorithms, you will need to think about further aspects of the project, for instance:
	- The type of ML technique to be used.
	- The task to be applied (for instance, classification, regression)
	- The evaluation and validation criteria to have your model accurately addressing the the questions I raised before.
 
We will use all the notebooks used in the previous weeks so that you can start your hands-on straight away! Have fun! 

## Pre-session Reading & Resources

We will be using Scikit-learn, Keras and TensorFlow. You can find the documentation of  Scikit-learn [here](http://scikit-learn.org/stable/). You can have a reading about Keras [here](https://keras.io). TensorFlow is an optimised library in python for dealing with tensors (for instance, weigh matrices in a neural network). TensorFlow documentation can be found [here](https://www.tensorflow.org). 

We will continue to use the book "Hands-on Machine Learning with Scikit-learn and Tensorflow" as the main reference. The collection of python notebooks related to the book with all the python implementations can be found [here](https://github.com/ageron/handson-ml.) They can be handy today!!  

Stack Overflow is a useful website to find solutions for when you get stuck with python and/or scikit-learn/keras/tensorflow: https://stackoverflow.com/questions/tagged/scikit-learn, https://stackoverflow.com/questions/tagged/keras, https://stackoverflow.com/questions/tagged/tensorflow. 

**Have a look at this flowchart from Scikit-learn!!** 

[This flowchart](http://scikit-learn.org/stable/tutorial/machine_learning_map/index.html) from Scikit-learn documentation gives you tips about how to find the right estimator for your problems, based on some questions about the data. You should have a look at it before starting your project!
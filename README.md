**Tip**: When opening .ipynb files if you get **`Sorry,... Reload?`** error in Github, use **[https://nbviewer.jupyter.org/](https://nbviewer.jupyter.org/)**  

# Classroom activities
* There will be 20+ classroom activities for you to practice and complete.
* You will need to complete at least 20 activities in order to receive full points. If you complete more than 20 activities, you will receive a 0.5 bonus point for each activity you complete.
* The first half set of activities are relatively easy (with Notebooks provided).

# Recommended datasets
* Classification and regression datasets at https://archive.ics.uci.edu/ml/datasets.php

# What to submit?
* A link to your Colab notebook (must be publicly accesible with the link) for each activity.

# Example
* Miguel Corona, a student who is taking this class, has made his activities notebooks publicly accessible through his Github repositories [here](https://github.com/mykon23/AI-2020/tree/master/Activities). You can refer to this to see how others are completing their activities. Thank you Miguel.

--------------  

## 1. Python
In this activity, the task is to learn how to use Google Colab and practice Python3. If you are doing Python programming for the first time, I suggest practicing Python at online platforms such as [codewars.org](https://www.codewars.com/) too.
* Lectures: [Google Colab](https://www.youtube.com/watch?v=PVsS9WtwVB8) and [Python3](https://www.youtube.com/watch?v=V42qfAPybp8)
* Notebooks: [Python3](../notebooks/python.ipynb)  

What to submit?   
* A link to your notebook where you practiced Python3.

## 2. Numpy, Matplotlib
In this activity, the task is to practice Numpy, Matplotlib, and Plotly.
* Lectures: [Numpy](https://www.youtube.com/watch?v=Omz8P8n-5gY) and [Matplotlib & Plotly](https://youtu.be/aIzkkjRzVdA)
* Notebooks: [Numpy](../notebooks/numpy.ipynb) and [Matplotlib & Plotly](../notebooks/matplotlib_plotly.ipynb)

What to submit?  
* A link to your notebook where you practiced Numpy, Matplotlib, and Plotly.

## 3. Basic data analysis using Pandas
In this activity, the goal is to learn how to use Pandas for basic data analysis. After learning the basics of Pandas from the resources below, the task is to repeat the steps for a different dataset of your choice, on a dataset other than the 'pima-diabetes' dataset.
* Notebooks: [Pandas](../notebooks/pandas.ipynb)

What to submit?  
* A link to your notebook where you performed data analysis using Pandas.

## 4. Univariate linear regression (Chapter 18)
In this activity, the goal is to practice univariate linear regression. When selecting variables (columns) for performing linear regression, it is important to choose continuous variables and not binary variables. Before feeding the data to the regression model, it is often important to normalize/standardarize your input dataset. You may need to normalize your data for regression to work. Here, the task is to perform univariate linear regression on a dataset of your choice (other than the 'pima-diabetes' dataset).  
* Lectures: [Univariate Linear Regression](https://youtu.be/yH7AUm2EHTM), [Data cleaning](https://youtu.be/0bj6KbEUJ_o), and [Data normalization](https://youtu.be/Tu8Dl3zorgg)

What to submit?  
* A link to your notebook where you performed univariate linear regression.

## 5. Linear regression with at least two input variables (Chapter 18)
In this activity, the goal is to practice linear regression on a dataset with more than one input variables. When selecting variables (columns) for performing linear regression, it is important to choose continous variables and not binary variables. Before feeding the data to the regression model, it is often important to normalize/standardarize your input dataset. You may need to normalize your data for regression to work. Here, the task is to perform linear regression on a dataset of your choice (other than the 'pima-diabetes' dataset).  
* Lectures: [Linear regression with two input variables](https://youtu.be/IOaif62O06k) and [Data normalization](https://youtu.be/Tu8Dl3zorgg)

What to submit?  
* A link to your notebook where you performed linear regression.

## 6. Logistic regression (Chapter 18)
In this activity, the goal is to practice logistic regression on a dataset with more than one input variables. When selecting variables (columns) for performing logistic regression, it is important to select a binary variable (i.e. the values of this variable must be 0 or 1, nothing else) as the output variable. Before feeding the data to the model, it is often important to normalize/standardarize your input dataset. You may need to normalize your data for classification to work. Here, the task is to perform logistic regression on a dataset of your choice (other than the 'pima-diabetes' dataset). 
* Lectures: [Logistic regression](https://youtu.be/KEYgPOcqmsw) and [Data normalization](https://youtu.be/Tu8Dl3zorgg)

What to submit?  
* A link to your notebook where you performed logistic regression.

## 7. Binary classification using NN
In this activity, the goal is to practice training a neural network model to perform binary classification. A neural network classifier should be more accurate than a basic logistic regression model. This is because a neural network model has more parameters (weights and biases) to learn the patterns in the data. A binary classifier can be evaluated using metrics such as accuracy, precision, and recall. Interpreting the accuracy of a binary classifier can be tricky. This is because the baseline accuracy, i.e., minimum accuracy, is at least 50%. A good classifier should result in an accuracy that is much higher than a baseline accuracy. The tasks in this activity are (i) Build a neural network classifier for a dataset of your choice, (ii) Evaluate your model using accuracy, precision, and recall, (iii) Compare the accuracy of your model with the baseline accuracy, and (iv) Compare the performance of the neural network with a logistic regression model.
* Lectures: [Binary classification](https://youtu.be/PM6uvCLyeXM)
* Articles: [A Visual and Interactive Guide to the Basics of Neural Networks](http://jalammar.github.io/visual-interactive-guide-basics-neural-networks/)

What to submit?  
* A link to your notebook along with your findings and discussions.

## 8. See [here](https://github.com/badriadhikari/AI-2020fall/blob/master/activities/Chapter_activities.md#8-breadth-first-search-bfs-tree-chapter-3).

## 9. See [here](https://github.com/badriadhikari/AI-2020fall/blob/master/activities/Chapter_activities.md#9-implement-the-breadth-first-search-bfs-algorithm-chapter-3).

## 10. Regression using NN and evaluation
In this activity, the goal is to practice training a neural network model to perform regression, i.e. predict continuous values. A neural network regression model should be more accurate than a basic linear regression model. This is because a neural network model has more parameters (weights and biases) to learn the patterns in the data. A regression model can be evaluated using metrics such as mean absolute error (MAE). This activity has five tasks: (i) Build a neural network regression model for a dataset of your choice, (ii) Evaluate your model using MAE, (iii) Compare the MAE of your model with a linear regression model, (iv) Assess if your model is biased towards predicting either larger values more correctly or smaller values more correctly, and (v) Experiment with various loss functions such as mae, mse, mean_squared_logarithmic_error, and logcosh, to find out which delivers the lowest MAE.
* Lecture: [Regression using neural networks](https://youtu.be/RG3QB7HGcVM)

What to submit?  
* A link to your notebook along with your findings and discussions.

## 11. Overfitting vs generalization
In this activity, the goal is to learn the concepts of overfitting, underfitting, generalization, and the purpose of splitting a dataset into training set and validation set. For a standard tabular classification dataset of your choice, where the output variable is a binary variable, the first step is to shuffle the rows (see example code below). The next step is to split the rows into training and validation set. For small datasets, selecting a random 30% of the rows as the validation set and leaving the rest as the training set works well. For larger datasets, smaller percents can be enough. This splitting yeilds four numpy arrays - XTRAIN, YTRAIN, XVALID, and YVALID (see example code below). For normalizing the data and to obtain the 'mean' and 'standard deviation' it is important to only use the XTRAIN array, not XVALID. XVALID should be normalized using the mean and standard deviation obtained from XTRAIN. Then the main question one should ask is - if a model is trained using the training data (XTRAIN and YTRAIN) how does it perform on the validation set (XVALID and YVALID)? In this activity there are two tasks: (i) Build a neural network model to overfit the training set (to get almost 100% accuracy or as high as it is possible) and then evalute on the validation set, and (ii) Evaluate the accuracy of the model for the training set and the validation set and discuss your findings. To obtain high accuracy on the training set, one can build a larger neural network (with more layers and more neurons per layer) and train as long as possible.
* Lectures: [Overfitting, generalization, and data splitting](https://youtu.be/1EfGsw-Szyg) 

```python
# Shuffle the datasets
import random
np.random.shuffle(dataset)
```

```python
# Split into training and validation, 30% validation set and 70% training 
index_30percent = int(0.3 * len(dataset[:, 0]))
print(index_30percent)
XVALID = dataset[:index_30percent, "all input columns"]
YVALID = dataset[:index_30percent, "output column"]
XTRAIN = dataset[index_30percent:, "all input columns"]
YTRAIN = dataset[index_30percent:, "output column"]
```

```python
# Learn the model from training set
model.fit(XTRAIN, YTRAIN, ...)
```
      
```python
# Evaluate on the training set (should deliver high accuracy)
P = model.predict(XTRAIN)
accuracy = model.evaluate(XTRAIN, YTRAIN)
```

```python
#Evaluate on the validation set
P = model.predict(XVALID)
accuracy = model.evaluate(XVALID, YVALID)
```

What to submit?  
* A link to your notebook where you have completed the tasks. The notebook should also have answers to the following questions: 
  1. Does your model perform better (in terms of accuracy) on the training set or validation set? Is this a problem? How to avoid this problem?
  1. Why can over training be a problem?
  1. What is the difference between generalization, overfitting, and underfitting?
  1. Why should you not normalize XVALID separately, i.e. why should we use the parameters from XTRAIN to normalize XVALID?  

## 12. See [here](https://github.com/badriadhikari/AI-2020fall/blob/master/activities/Chapter_activities.md#12-alpha-beta-pruning-chapter-5).
 
## 13. Learning curves
This activity assumes that you have successfully completed all previous activities. It also requires some focus. Learning curves are a key to debug and diagnose a model's performance. The goal in this activity is to plot learning curves and to interpret various learning curves. For a regression dataset of your choice, the first step is to shuffle the dataset. The next step is to split the dataset into the four arrays: XTRAIN, YTRAIN, XVALID, and YVALID. The next step is to train a neural network model using `model.fit()`. However, this time, XVALID and YVALID will also be passed as arguments to the `model.fit()` method. This is so when we call the method, it can evaluate the model on the validation set at the end of each epoch (see code block below). It is extremely important to understand that the `model.fit()` method does NOT use the validation dataset to perform the learning, it is only to evaluate the model after each epoch. When calling the `model.fit()` method we can also save its output in a variable, usually named `history`. This variable can be used to plot learning curves (see code block below). The task in this activity is to plot many learning curves in various scenarios. In particular, it is of intererest to observe and analyze how the learning plots look like various settings. The following article discusses learning curves in more detail.
* Lectures: [Some insights on learning curves](https://youtu.be/seRETv52U1Q)
* Articles: [Learning curves for diagnosing machine learning model performance](https://machinelearningmastery.com/learning-curves-for-diagnosing-machine-learning-model-performance/)

```python
# Do the training (specify the validation set as well)
history = model.fit(XTRAIN, YTRAIN, validation_data = (XVALID, YVALID), verbose = 1)
# Check what's in the history
print(history.params)
# Plot the learning curves (loss/accuracy/MAE)
plt.plot(history.history['loss']) # replace with accuracy/MAE
plt.plot(history.history['val_accuracy']) # replace with val_accuracy, etc.
plt.ylabel('Accuracy')
plt.xlabel('epoch')
plt.legend(['training data', 'validation data'], loc='lower right')
plt.show()
```

What to submit?  
* A link to your notebook where you produce the learning curves that represent the following cases:
   1. the validation set is too small relative to the training set - for example, only 1% or 2% of the total rows of data.
   1. the training set is too small compared to the validation sat - for example, only 1% or 2% of the total rows of data.
   1. a good learning curve (practically good)
   1. an overfitting model
   1. a model that shows that further training is required
   1. an underfit model that does not have sufficient capacity (also may imply that the data itself is difficult)

## 14. Fine-tuning hyper-parameters of your model
In this activity, the task is to learn how to design and train a model that does well on the unseen (validation) daset. The weights and biases of a neural network model are its parameters. The parameters such as the number of layers of neurons, numbers of neurons in each layer, number of epochs, batch size, activation functions, choice of optimizer, choice of loss function, etc. are the hyperparameters of a model. When training a model for a new dataset an extremely important question is - what combinations of hyperameters yield the maximum accuracy on the validation set? Remember, when playing with activation functions, the activation of the last layer should not change - it should always be sigmoid for binary classification and ReLU or linear for regression. The task is in this activity is to try as many hyperparameters as possible to obtain the highest possible accuracy on the validation set. For a **classification dataset of your choice**, the first step is to create a notebook where you can train the model using the training set and evaluate on the validation set. Then, the objective is to find the optimal (best) hyper-parameters that maximize the accuracy (or minimize MAE) on the validation set. 

* Lectures: [What are hyperparameters and why are they important](https://youtu.be/ggCYYgx2MNM) 

Below are the hyperparameters to optimize:
1. The number of layers in the neural network (try 1, 2, 4, 8, 16, etc.).
1. The number of neurons in each layer (try 2, 4, 8, 16, 32, 64, 128, 256, 512, etc.).
1. Various batch sizes (8, 16, 32, 64, etc.).
1. Various number of epochs (2, 4, 8, ..., 5000, etc.).
1. Various optimizers (rmsprop, sgd, nadam, adam, gd, etc.)
1. Various activation functions for the intermediate layers (relu, sigmoid, elu, etc.)

What to submit?  
* A link to your notebook with all the experiments.

## 15. See [here](https://github.com/badriadhikari/AI-2020fall/blob/master/activities/Chapter_activities.md#15-implement-bm25-function-chapter-22).

## 16. See [here](https://github.com/badriadhikari/AI-2020fall/blob/master/activities/Chapter_activities.md#16-implement-pagerank-algorithm-chapter-22).

## 17. Early stopping
Assumption: You already know (tentatively) what hyperparameters are good for your dataset
* Find a regression dataset of your choice and split into training and validation set
* There are two objectives in this activity:  
  a. Implement automatic stopping of training if the accuracy does not improve for certain epochs  
  b. Implement automatic saving of the best model (best on the validation set)  
* Define callbacks as follows (and fix the obvious bugs):
  ```python
  from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
  # File name must be in quotes
  callback_a = ModelCheckpoint(filepath = your_model.hdf5, monitor='val_loss', save_best_only = True, save_weights_only = True, verbose = 1)
  # The patience value can be 10, 20, 100, etc. depending on when your model starts to overfit
  callback_b = EarlyStopping(monitor='val_loss', mode='min', patience=your_patience_value, verbose=1)
  ```
* Update your `model.fit()` by adding the callbacks:
  ```python
  history = model.fit(XTRAIN, YTRAIN, validation_data=(XVALID, YVALID), epochs=?, batch_size=?, callbacks = [callback_a, callback_b])
  ```
* Before you evaluate your model on the validation set, it is important to load the "checkpoint-ed" model:
  ```python
  # File name must be in quotes
  model.load_weights(your_model.hdf5)
  ```
* Plot the learning curves and demonstrate that model checkpointing helps to obtain higher accuracy on the validation set
  <img src="model-checkpoint.png" align="middle" width="600"/>
* At the end of your notebook, answer the following questions:  
  a. Almost always, training with early stopping finishes faster (because it stops early). Approximately, how long does it take for your training to finish with and without early stopping?  
  b. When model checkpointing, your checkpointed model will almost always be more accurate on the validation set. What is the MAE on the Validation set with and without model checkpointing?

## 18. See [here](https://github.com/badriadhikari/AI-2020fall/blob/master/activities/Chapter_activities.md#18-implement-convolution-operation-chapter-24).

## 19. Iterative feature removal & selection
* As of now, it is assumed that given a dataset (of your choice) you can build a model that can do reasonably well on the validation set, i.e. you have a good idea of the network architecture needed, the number of epochs needed, model Checkpointing, the approximate MAE or accuracy that one might expect, etc.
* Here we will train a model using the training set and evaluate on the validation set; you are free to choose your own dataset (even your project dataset is fine)
* In this activity you will implement a simple Recursive Feature Elimination (RFE) technique to remove redundant or insignificant input features
* Expected output 1: Plot the significance (importance) of each feature after training your model using one feature at a time:   
   a. X-axis represents the feature that was used as the input  
   b. Y-axis is accuracy or MAE of the validation set  
   <img src="feature_importance.png" align="middle" width="450" border="2"/>
* Observing these MAE/accuracy values, we can rank the features by their importance (how informative each one is)
* Next, iteratively remove one feature at a time (starting with the least significant feature) and repeat the training noting the accuracy/MAE on the validation set
* Expected output 2: Plot to report your findings:   
   a. X-axis represents feature removal, for example, second entry is after removing feature1, and third entry is after removing feature1 and feature2  
   b. Y-axis is accuracy or MAE of the validation set  
   <img src="feature_removal.png" align="middle" width="550" border="2"/>  

## 20. Binary classification and regression using XGBoost
* The goal here is to study how the XGBoost library can be used for building boosted decision trees for classification and regression   
   <img src="xgboost.png" align="middle" width="800" border="1"/>  
* Task 1: Binary classification using XGBoost
   1. Study how the xgboost library can be used for binary classification - [notebook](../notebooks/XGBoost_for_classification.ipynb)
   1. For a dataset of your choice, build an `XGBClassifier()` model to obtain higest possible accuracy on the validation set (by adapting the hyperparameters of the model such as `n_estimators` and `max_depth`)
   1. Visualize at least one decision tree and write a few sentences interpreting the decision tree
   1. Compare the accuracy/precision of this model with a neural network model
   1. Expected output: A notebook that has answers for all the four tasks above **(on a different dataset)**

* Task 2: Regression using XGBoost
   1. Study how the xgboost library can be used for regression - [notebook](../notebooks/XGBoost_for_regression.ipynb)
   1. For a dataset of your choice, build an `XGBRegressor()` model to obtain lowest possible MAE on the validation set (by adapting the hyperparameters of the model such as `n_estimators` and `max_depth`)
   1. Visualize at least one decision tree and write a few sentences interpreting the decision tree
   1. Compare the MAE of this model with a neural network model
   1. Expected output: A notebook that has answers for all the four tasks above **(on a different dataset)**

## 21. Feature importance and removal using XGBoost
* In this activity, you will study feature importance using the XGBoost library
* After building your XGBoost model (classifier or regressor) studying feature importance is just one line of code
   ```python
   from xgboost import plot_importance
   plt.rcParams['figure.figsize'] = [8, 4]
   plot_importance(xg_cla)
   ```
   <img src="xgboost-feature-importance.png" align="middle" width="400" border="1"/>  
* Expected outputs:
   1. Plot feature importance for a dataset of your choice and briefly discuss if the findings are similar or different from your intuition (i.e. why does the feature importance makes sense?)
   1. Iteratively remove one feature at a time (starting with the least significant feature) and repeat the training noting the accuracy/MAE on the validation set and plot to report your findings (X-axis represents feature removal, for example, second entry is after removing feature1, and third entry is after removing feature1 and feature2, etc.)
   1. Discuss how your findings relate to your feature importance and reduction study using neural networks

## 22. Learning with missing values & noisy data
* In this activity, we will investigate the impact of "amount of data" and missing/noisy data
* For a dataset of your choice, randomly set random rows/columns (around 10% of your total rows) to non-standard values such as -9999 or 9999 and repeat your training/evaluation.
  * Expected output: Your discussion of how noisy data impacts the accuracy/MAE on the validation set
  ```python
  # Sample code to make data noisy
  import numpy as np
  dataset = np.loadtxt('winequality-red.csv', delimiter=",", skiprows=1)
  for i in range(100):
      # Choose a random row
      rand_row = random.randint(0, len(dataset) - 1)
      # Choose a random column (except the last/output column)
      rand_col = random.randint(0, len(dataset[0, :]) - 2)
      print(rand_row, rand_col)
      # Set the row and column to -9999 or 9999
      dataset[rand_row, rand_col] = 9999
  ```
* For a dataset of your choice, iteratively decrease the total number of data (rows) and and evaluate the accuracy/MAE on the validation set - please do not change the validation set (keep the same number of rows in each run); only decrease the number of rows in the training set.
  * Expected output: A plot showing how the # of rows (x-axis) impacts the accuracy/MAE on validation data (y-axis) - with at least 8/10 points on the plot (for example: 1%, 2%, 5%, 10%, 20%, 40%, 60%, and 80%)

## 23. Cross-validation
* In this activity, we will study cross-validation   
   <img src="cross-val.png" align="middle" width="500" border="1"/>  
* Here is a sample Python code:
   ```python
   # Say our data has 100 rows and 12 columns - last column is the output column 
   print(dataset.shape)
   X = dataset[ : , :-1]
   Y = dataset[ : , -1]
   ```
   
   ```
   (100, 12)
   ```

   ```python
   from sklearn.model_selection import KFold
   # Define 5 folds
   kf = KFold(n_splits = 5)
   # Returns the number of splitting iterations in the cross-validator
   kf.get_n_splits(X)
   print(kf)
   ```
   
   ```
   KFold(n_splits=5, random_state=None, shuffle=False)
   ```
   
   ```python
   split = 0
   for train_indices, valid_indices in kf.split(X):
       print('')
       split += 1
       print('Split:', split)
       print('TRAIN:')
       print(train_indices.shape)
       print(train_indices) 
       print('VALID:')
       print(valid_indices.shape)
       print(valid_indices)
       XTRAIN, XVALID = X[train_indices], X[valid_indices]
       YTRAIN, YVALID = Y[train_indices], Y[valid_indices] 
       # Your code for training/evaluation goes here..
       # ToDo..

   ```

   ```
   Split: 1
   TRAIN:    
   (80,)    
   [20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43   
     44 45 46 47 48 49 50 51 52 53 54 55 56 57 58 59 60 61 62 63 64 65 66 67   
     68 69 70 71 72 73 74 75 76 77 78 79 80 81 82 83 84 85 86 87 88 89 90 91   
     92 93 94 95 96 97 98 99]    
   VALID:   
   (20,)   
   [ 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19]
   ```
* Expected output: Create a table summarizing your accuracy/MAE in each split; here is a sample table:   

   | Split | Accuracy/MAE |
   |---|---| 
   | 1  | ?  |
   | 2  |  ? |
   | ...  | ... |



# Time series FAANG prediction using ANN

Sklearn implementaiton of Artificial Neural Network prediction of FAANG trading data

### 1. Folder Structure

```python
Repository
 |-- ANN_tuning_code.ipynb	# Main code notebook
 |    |-- input     # input csv data
 |    |    |-- fundamentals.csv              # 
 |    |    |-- prices-split-adjusted.csv     #
 |    |    |-- prices.csv                    # 
 |    |    |-- securities.csv                #
```

### 2. Overview

Sklearn implementaiton of Artificial Neural Network prediction of FAANG trading data

### 3. Data Source

This data was obtained through Kaggle: [link](https://www.kaggle.com/dgawlik/nyse). The dataset contained daily Open, Close, Low, High and Volume metrics for each stock on the NYSE from 2013 to 2016.

### 4. Methodology

The basic structure of an artificial neural network consists of 3 layers – input layer, layer of neurons and output layer which generates the predicted values.
 
The input values are first transformed by adding a weighting. This weight vector is initialised by the Keras function kernel_initializer and then passed through an activation function which decides whether to keep the neuron for the next network layer or output layer. The basic weight function is shown below applied on input value x. z=w^T x + b

Within the output neuron a cost function is calculated and minimised through an optimisation algorithm which in this project is minimising the MSE between the train stock price and predicted stock price from the cross-validation holdout set. Once the optimisation converges the error is fed back into the input layer and the weights are updated. Common optimisation algorithms are stochastic gradient descent and adaptive moment estimation (Adam).

Once the entire learning process has been completed which is determined by the number of epochs or forward-backward passes through the neural network layers, the output layer produces a final estimated stock price.

To reduce the risk of overfitting, a dropout term is used. This sets a percentage of samples to zero after the neuron layer. In my hyperparameter tuning I selected 10% of samples to be removed.

ANN model was chosen as it has been demonstrated to be a strong candidate model for financial time series analysis and it can be customised to be very simple or complex through the number of choice of number of neurons and the transformations involved at the end of each layer. This means it can capture both simple and complex patterns that are common in a selection of stocks.

The pre-processing methods used on the train dataset was standardisation which was important to enable convergence of the ANN weights.

Further the close price was shifted 1 row so that the open, low, high prices etc were corresponding to the following days close price (today + 1).


### 5. Results

The best model resulting from the tuning process was a single layer ANN with a normal weight initialisation, 100 neurons and a small learning rate which is more likely to find the minimal MSE point for a specific neuron. This configuration was used in cross validation testing on the FAANG stocks.


The large cross validation scores for Amazon and Netflix could be the result of correlated predictors or explainable due to the adaptive Adam estimation finding a saddle point along the gradient descent. The best performing stock using the ANN model was Netflix with a 88% cross validation score. Compared to the random forest the ANN was computationally slow which might be due to the dropout function.

As seen from the graphs below, visually Facebook's prediction had the highest error (blue = actual, orange = predicted) with a general observation being the ANN was effective at predicting the direction of the daily stock movement but not the magnitude.

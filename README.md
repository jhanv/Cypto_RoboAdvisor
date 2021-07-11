# Cypto_RoboAdvisor

## 1 Problem

The central problem of this paper is to analyse the cryptomarket  and see if a deep learning model can predict the fluctuation of prices in conjunction with sentiment data. To this end, we hope to construct a few models of varying complexity to help learn the patterns (if any) and help eliminate some of the uncertainty of this market. We believe that crypto currencies would have a prominent position in the future of finance and this paper would shed light into how people view the currency.  

## 2 Introduction

It is well known that the market is a volatile beast. The crypto market is specifically more volatile and often guided by public sentiment. 
Our interest in cryptocurrencies stem from the limitation of conventional currencies. Conventional credit and cast is often used to make transactions but it is overly controlled and only sustains small transfers of money. To this end, we believe cryptocurrencies have an important role to play in our future. 
Researches have long been interested in predicting the prices of cryptocurrencies using simple models like linear regression and ARIMA. In our project we tried three models of differing complexities to see how well it would do. The models we used are: MLP, Transformers and LSTM.
For the data we have used prices from different exchanges over the years and additional data from google trends, reddit and twitter. 

## 3 Data Engineering

### 3.1 Data Extraction

For historical bitcoin price data, we extracted bitcoin’s historical daily trading volume, and open, close, high, and low prices from the Bitstamp exchange. For daily sentiment data, we extracted for that day the number of Reddit that mentioned bitcoin, total number of likes on bitcoin related Reddit posts, total number of Twitter posts mentioning bitcoin, Google interest index value for the bitcoin keyword, and a weighted sentiment score on Reddit posts mentioning bitcoin. 


The weighted sentiment score was calculated by passing each Reddit post mentioning bitcoin form that day into a pre-trained Hugging Face transformer sentiment analyzer model, and then taking a weighted average of the model’s output sentiment scores based on the popularity of that post. This transformer model outputted a positive or negative label, along with a confidence score that we used as the numerical value of the sentiment score. Although this model was pre-trained on text that did not include Reddit posts, we manually looked through a good number of the sentiment labels to make sure they accurately reflected the post. 


### 3.2 Data hyperparameters 

Each of the models was tested across various data-specific hyperparameters and model-specific hyperparameters. The data-specific hyperparameters were common across all sections, and are defined here. 

Data format: Different transformations were used on the training data before passing it into the models.  Using “absolute data” for a model refers to using the raw, absolute pricing and sentiment data numbers as training data. Using “percentage data” refers to extracting day to day percentage changes in the pricing and sentiment data numbers as training data. Using “difference data” refers to extracting day to day absolute differences in the pricing and sentiment data numbers as training data. 
Normalization technique: Different normalization techniques were used on the training data post transformations. These techniques included 1) min max scaling, which scales the data on a 0-1 scale, 2) standard scaling, which centers the data around a mean of 0 and standard deviation of 1, and 3) the lack of any normalization. 
Sequence length: The number of days of historical price/sentiment data that is used to predict the future price. In other words, if sequence length is 14, then each training data item contains 14 days of historical price/sentiment data, and is used to output one value for the future price. 
Prediction day: The number of days in the future that the model attempts to predict the price for. 


### 3.3 Data splitting 

For every model, we split the data into 80%, 10%, and 10% for the training, validation, and testing datasets, respectively. The time series data is not shuffled before splitting it into the training, validation, and testing datasets. In other words, the data from 2014 to around mid 2018 was used for training, mid 2018 to early 2019 for validation, and early to late 2019 for testing. This split up will best test whether historical correlations will carry over to the future. Shuffling the data would have probably led to artificially inflated testing results. 

## 4 Models

We experimented with three models, MLP, LSTM and Transformers

### 4.1 MLP MODEL

The first model that we tried was the simplest model option: a MLP model. Since a MLP model does not have any inbuilt time-series attention mechanism, it is the simplest model that could be used for our task. After experimentation and hypertuning, we decided on using ReLU activation, MSE loss, Adam optimization, and a learning rate of around 3e-5. 

### 4.2 TRANSFORMER:

The next model under consideration was a transformer model. A transformer model is a neural network that uses self-attention to give higher weights to the relevant parts of the time series data that make it easier for the prediction. We will be using a Multi head attention layer which will consist of Single head attention layer all constructed using Keras from Tensorflow.

As for the input data we first use a time2vector which will be discussed in the following sections, which is then concatenated to the actual data. The transformer also has multiple dropouts and batch normalization. After the self attention layer, the data is passed through a few conv1d layers to output a value of the required size.

#### 4.2.1 Architecture

Time2vector:
One of the more novel changes to these transformers is that the ordering of the different days in the sequence is encoded in two extra fields: periodic and non-periodic time features. The equation to derive these two feature are as below:



The intuition behind this feature is that the non-periodic feature captures the movement in time whereas the periodic feature holds the information about the seasonality in the data. This function is performed over the mean of every other feature in each instance of the sequence.

Self Attention heads

The attention heads are very similar to the ones commonly used in most literature for NLP models. We have three inputs: Query, Key , Value. All three of these inputs are the sequence itself, given that we are in need of a self attention head.

Our multi head attention is a list of single attention heads. The outputs of these heads are then concatenated before passing it through a fully connected layer


Transformer Encoder layer

The transformer encoder layer gets its input from the previous attention heads. The inputs are then run through a few Conv1D layers with kernel size 1. This effectively makes it a fully connected layer. There are “ReLU” activations between these layers. There are dropouts (15%) and batch normalizations as well that only function in the forward pass.  


#### 4.2.2 Training

The complete data while training included the difference time-series data for price (absolute), the differential price and the sentiment scores (reddit, google trends, twitter). The training was done with both the entire data as well as part of the data to see how well it would learn. It was also found that using a standard scaler gave better results so all the training for this model was done with such a normalization.

Training was done for 35 epochs which takes around 2-5minutes.

### 4.3 LONG SHORT TERM MEMORY MODEL (LSTM):

LSTM models are chosen since they are powerful sequence predictions because they are able to store past data. The assumption was that LSTM would be better at extracting the long term patterns that predict the price. 

#### 4.3.1 Architecture

The architecture was made with the help of Keras. After experimenting with a different number of LSTM layers, it was found that 2 layers were optimal for our purposes. Using “ReLU” activation in between the layers seemed to worsen performance. A dropout layer of 15% was added in between the layer for just the forward pass. There is finally a dense layer at the end to shape the output in the format we require.

#### 4.3.2 Training

The training for the again consists of the whole dataset used for the transformer with different runs having complete and partial data.

The model was run for 50 epochs and the losses converged quickly. Mean Squared Error was the loss function used, with Adam Optimizer and Metrics as Mean Absolute Error and Mean Absolute Percentage Error


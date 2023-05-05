# Comment Sentiment Analysis

Sentiment analysis is a technique used in natural language processing and machine learning that involves identifying the emotional tone or attitude expressed in a piece of text, such as a tweet, product review, or news article. The goal of this project is to determine whether the sentiment expressed in the text is positive, negative, or neutral.

### Data
The dataset was taken From Kaggle "Sentiment Analysis Dataset", you can find by the following link https://www.kaggle.com/datasets/abhi8923shriv/sentiment-analysis-dataset?select=training.1600000.processed.noemoticon.csv. 

Data file format has 6 fields:
- 0 - the polarity of the tweet (0 = negative, 2 = neutral, 4 = positive)
- 1 - the id of the tweet (2087)
- 2 - the date of the tweet (Sat May 16 23:58:44 UTC 2009)
- 3 - the query (lyx). If there is no query, then this value is NO_QUERY.
- 4 - the user that tweeted (robotickilldozr)
- 5 - the text of the tweet (Lyx is cool)

### RNN model
For this taks, we first define a Sequential model and add layers to it. The first layer is an Embedding layer, which maps each word in our vocabulary to a vector of length 100. The input_dim argument specifies the size of our vocabulary, while input_length specifies the length of each input sequence (i.e., the number of words in each document).

Next, we add two LSTM layers with 128 and 64 units, respectively, followed by Dropout layers to prevent overfitting. Finally, we add a Dense output layer with a single unit and a sigmoid activation function, which outputs a probability between 0 and 1 indicating the likelihood that the input belongs to the positive class.

We then compile the model with binary cross-entropy loss and the Adam optimizer, and specify that we want to track the accuracy metric during training. Finally, we fit the model to our training data, specifying a validation set, the number of epochs to train for, and the batch size to use during training

# XGBoost-Stock-Classifier

Overview: 

The goal of this project was to use machine learning techniques to accurate predict if a stock will move up or down in price within the following hour. The model selected was XGBoost as the decision tree model can handle and predict time series data exceptionally well. Special thanks to Yahoo as the yFinance stock data set provided the data necessary for training the model. The model performed exceptionally well on 5 days of apple stock data as the Accuracy, through finetuning, was eventually able to surpass 93%. Given the volatitlity of the market this is a huge win since it is very rare for a regualr stock broker to match this prediction accuracy. On top of this Therefore, this application can serve as stock advisor to the average consumer while also being a great tool that stock brokers can fine tune for their individual prediction needs. 

Limiting variables and ways to improve:

One of the main reasons this model was not used in testing for long term stocks is due to insufficient compute power to train a model with that much data. However, an entitiy such a trading firm could provide such necessary compute power. Moreover, more experienced stock brokers could train the model with better features to increase short term and long term accuracy.

Future works: 

In the future, a CNN (Convolutional Neural Network) will be added to analize the past hour of time series data for each stock price to aid the XGBoost algorithim in its prediction (curently in dev). Also, this model will be run using much more data in the near future, to test long term application, once I have access to more compute.




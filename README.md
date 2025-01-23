# XGBoost-Stock-Classifier

Overview: 

The goal of this project was to use machine learning techniques to accurate predict if a stock will move up or down in price within the following hour. The model selected was XGBoost as the decision tree model can handle and predict time series data exceptionally well. Special thanks to Yahoo as the yFinance stock data set provided the data necessary for training the model. The model performed exceptionally well on 5 days of apple stock data as the Accuracy, through finetuning, was eventually able to surpass 93%. Given the volatility of the market this is a huge win since it is very rare for a regular stock broker to match this prediction accuracy. On top of this, the application can serve as stock advisor to the average consumer while also being a great tool that stock brokers can fine tune for their individual prediction needs. 

Limiting variables and ways to improve:

One of the main reasons this model was not used in testing for long term stocks is due to insufficient compute power to train a model with that much data. However, an entitiy such a trading firm could provide such necessary compute power. Moreover, more experienced stock brokers could train the model with better features to increase short term and long term accuracy.

Future works: 

I would like to add more data inorder to increase the accuracy further and potentially use the same framework to train a model for long term stock prediction abe to make predictions a couple months in advance




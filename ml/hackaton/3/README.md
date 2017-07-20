

> Abstract: According to the SIP RFC a VOIP call can return standard error code 2xx, 4xx, and so on. Over a 5 minutes window, the distribution follows a standard distribution. A model will be trained to predict normal and abnormal distributions. The last 24 hours are relevant, data older than 24 hours should be forgotten by the model.

Data:
  * Input: Per minute, each input is a N dimensional vector X[i] = %(error(ixx)). Range: [0 - 1], and sum(X[i]) = 1.

Expected (predicted) output: timestamp, anomaly label, and label probabilities for each data point

Hints:
  * RNN (Recurrent Neural Network) are best suited to process time series data



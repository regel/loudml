
> Abstract: the mean opinion score (MOS) is a common metric used in VOIP quality estimation. The score ranges from 1 to 5. The average MOS is measured with 1 minute resolution. A model will be trained to predict normal and abnormal low average values. 

Data:
  * Input: average(MOS) per minute i.e. 1 dimensional data

Expected (predicted) output: timestamp, anomaly label, and label probabilities for each data point

Hints:
  * RNN (Recurrent Neural Network) are best suited to process time series data. A worldwide classic is how text prediction models are trained by shifting the text by one character



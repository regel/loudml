
# NN-SOM

The examples are based on MNIST and random color sampling.

However the principle is general and applies to N dimension vectors.
As in SUNSHINE research paper suggest, we can try to apply the
technique to VOIP CDR and call history profiles in order to detect
fraud.

Data: randomly created using voip simu in regel/voip repo
  * input.json: contains call statistics for 100 accounts
  * fraud.json: same, but there is superimposed fraud in 1 customer account

Files:
  * ongjia.py : SOM example in Python by Ong Jia Rui [Available on Github](https://github.com/jrios6/Math-of-Intelligence/tree/master/4-Self-Organizing-Maps)
  * sachin.py : SOM example in Python by Sachin Joglekar. See his [blog](https://codesachin.wordpress.com/2015/11/28/self-organizing-maps-with-googles-tensorflow/)


import tensorflow as tf
import tensorflow.contrib.keras.api.keras.models
from tensorflow.contrib.keras.api.keras.models import model_from_json


def init(): 
	json_file = open('model/model.json','r')
	loaded_model_json = json_file.read()
	json_file.close()
	loaded_model = model_from_json(loaded_model_json)
	#load woeights into new model
	loaded_model.load_weights("model/model.h5")
	print("Loaded Model from disk")

	#compile and evaluate loaded model
	loaded_model.compile(loss='mean_squared_error', optimizer='adam')
	graph = tf.get_default_graph()

	return loaded_model,graph



from keras.layers import Layer, deserialize, serialize
from keras.models import Model
from keras.utils import CustomObjectScope
from keras.models import model_from_json
import os
import json

def save_model(model, filename):
    model_json = model.to_json()
    filename = '{}.json'.format(filename)
    with open(filename, "w") as json_file:
        json_file.write(model_json)
        json_file.close()


def load_model(filename, custom_layers=None):
    filename = '{}.json'.format(filename)
    assert os.path.exists(filename)
    model = None
    json_file = open(filename, "r")
    read_model = json_file.read()
    json_file.close()
    if custom_layers==None:
        model = model_from_json(read_model)
    else:
        with CustomObjectScope(custom_layers):
            model = model_from_json(read_model)
    return model
    
def save_layer(layer, path):
	
	fp = path + '.json'
	
	with open(fp, 'w') as json_file:
		json.dump(serialize(layer), json_file)
		

def load_layer(path, custom_layers={}):
    
    fp = path + '.json'
    config = {}
    layer = None
    
    with open(fp) as json_file:
        config = json.load(json_file)
        
    with CustomObjectScope(custom_layers):
        layer = deserialize(config)

    return layer
    

def set_model_trainable(model, trainable):
	model.trainable = trainable
	for layer in model.layers:
		layer.trainable = trainable
		if isinstance(layer, Layer):
			continue
		else:
			set_model_trainable(layer,trainable)





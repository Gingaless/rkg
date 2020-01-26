from keras.layers import Add, Input
from keras.models import Model
import keras.backend as K


class WeightedSum(Add):
	
	def __init__(self, alpha=0.0, **kwargs):
		
		super(WeightedSum, self).__init__(**kwargs)
		self.alpha = K.variable(alpha, name = 'ws_alpha')
		
		
	def _merge_function(self, inputs):
		
		assert len(inputs) == 2
		
		output = (1.0 - self.alpha)*inputs[0] + self.alpha*inputs[1]
		return output
		


if __name__=='__main__':
	
	inp1 = Input(shape=[1])
	inp2 = Input(shape=[1])
	
	m = Model(inputs=[inp1, inp2], outputs=WeightedSum(0.6)([inp1, inp2]))
	print(m.predict([[[1],[2]], [[3],[4]]]))
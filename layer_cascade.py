from models.layers import PoolLayer_inner,DenseNetLayer,ConvLayer, FCLayer,PoolLayer_blocks, PoolLayer, get_layer_by_name,ConvLayer_blocks
import tensorflow as tf


class LayerCascade:
	def __init__(self, _id, layers):
		self._id = _id
		self.layers = layers
		self.output_op = None
	
	@property
	def id(self):
		return self._id
	
	@id.setter
	def id(self, value):
		self._id = value
	
	@property
	def out_features_dim(self):
		for layer in self.layers[::-1]:
			if isinstance(layer, ConvLayer):
				return layer.filter_num
			elif isinstance(layer, FCLayer):
				return layer.units
		return None
	
	@property
	def depth(self):
		depth = 0
		for layer in self.layers:
			if isinstance(layer, ConvLayer) or isinstance(layer, FCLayer):
				depth += 1
		return depth
	
	def get_str(self):
		layers_str = [layer.layer_str for layer in self.layers]
		return '-'.join(layers_str)
	
	def build(self, _input, densenet, store_output_op=False):
		output = _input
		output=self.layers[0].build(output, densenet, store_output_op=store_output_op)
		#output=[ output[:,:,:,_i*self.number_of_base_channel:(_i+1)*self.number_of_base_channel] for _i in range(self.depth_)]

		index=0
		with tf.variable_scope(self._id):
			for layer in self.layers[1:]:
				
				output=layer.build(output, densenet, store_output_op=store_output_op)
				if isinstance(output,list):
						for i in output:
								if i is not None:
									print(i.get_shape(),end=" ")
								else:
									print(None,end=" ")
						print()
				else:
					print(output.get_shape())
					output=output

		if store_output_op:
			self.output_op = output
		return output
	
	def get_config(self):
		return {
			'_id': self._id,
			'layers': [layer.get_config() for layer in self.layers]
			

		}
	
	def renew_init(self, densenet):
		return {
			'_id': self._id,
			'layers': [layer.renew_init(densenet) for layer in self.layers]
		}
	
	def copy(self):
		return self.set_from_config(self.get_config(), init=self.renew_init(None))
	
	@staticmethod
	def set_from_config(config_json, init=None, return_class=True):
		_id = config_json['_id']
		'''number_of_block=config_json['number_of_block']
		number_of_base_channel=config_json['number_of_base_channel']
		depth=config_json['depth']'''
		layers = []
		
		import numpy as np
		
		#'''
		
		
		conv_i=0
		n_layers=2
		k=48
		input_filter_num=k*2
		layers.append(ConvLayer("conv_adjust",k*2))
		for i in range(3):
			for j in range(n_layers):
				layers.append(DenseNetLayer("DenseNetLayer_"+str(i*n_layers+j),k,4,3))
				#input_filter_num=input_filter_num+k
			if i<2:
				input_filter_num=(input_filter_num+k*n_layers)
				#layers.append(PoolLayer_inner("PoolLayer_"+str(i),input_filter_num,'avg'))
				layers.append(ConvLayer("conv_"+str(i),input_filter_num,1,1))
				layers.append(PoolLayer("pool_"+str(i),'avg',2))
				#append(ConvLayer_blocks("ConvLayer_blocks_"+str(i),12,(i+1)*11+1,2,2))
				#
		layers.append(PoolLayer("pool_"+str(i),'avg',8,use_bn=True, activation='relu'))
		layers.append(FCLayer("fc_0",10))

		#_id, _type,number_of_block=4, kernel_size=2
		'''
		for _i, layer_config in enumerate(config_json['layers']):
			layer_init = init['layers'][_i] if init is not None else None
			#print(layer_config['name'])
			layer = get_layer_by_name(layer_config['name'])
			layers.append(layer.set_from_config(layer_config, layer_init))
		#'''
		if return_class:
			return LayerCascade(_id, layers)
		else:
			return _id, layers


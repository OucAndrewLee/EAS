from models.model import BasicModel
import tensorflow as tf
import numpy as np
import copy
import random
import re
def apply_noise(weights, noise_config):
	if noise_config is None:
		return weights
	noise_type = noise_config.get('type', 'normal')
	if noise_type == 'normal':
		ratio = noise_config.get('ratio', 1e-3)
		std = np.std(weights)
		noise = np.random.normal(0, std * ratio, size=weights.shape)
	elif noise_type == 'uniform':
		ratio = noise_config.get('ratio', 1e-3)
		mean, _max = np.mean(weights), np.max(weights)
		width = (_max - mean) * ratio
		noise = np.random.uniform(-width, width, size=weights.shape)
	else:
		raise NotImplementedError
	return weights + noise


def get_layer_by_name(name):
	if name == 'conv':
		return ConvLayer
	elif name == 'DenseNetLayer':
		return DenseNetLayer
	elif name == 'fc':
		return FCLayer
	elif name == 'pool':
		return PoolLayer
	else:
		raise ValueError('Unknown layer type: %s' % name)




class BaseLayer:
	"""
	_id, batch normalization, activation, dropout, ready
	"""
	def __init__(self, _id, use_bn=True, activation='relu', keep_prob=1.0, ready=True, pre_activation=True):
		self._id = _id
		self.use_bn = use_bn
		self.activation = activation
		self.keep_prob = keep_prob
		self.ready = ready
		self.pre_activation = pre_activation
		
		self._scope = None
		self._init = None
		self.output_op = None
	
	@property
	def id(self): return self._id
	
	@id.setter
	def id(self, value): self._id = value
	
	@property
	def init(self):
		return self._init
	
	@property
	def param_initializer(self):
		if self._init is None:
			return None
		import numpy as np
		
		param_initializer = {}
		for key in self.variable_list.keys():
			if self._init[key] is not None:
				param_initializer[key] = tf.constant_initializer(self._init[key])
				
		if len(param_initializer) == 0:
			param_initializer = None
		return param_initializer
	
	def renew_init(self, net: BasicModel):
		if net is None:
			return copy.deepcopy(self._init)
		
		self._init = {}
		for key, var_name in self.variable_list.items():

			var = net.graph.get_tensor_by_name('%s/%s' % (self._scope, var_name))
			self._init[key] = net.sess.run(var)
		if len(self._init) == 0:
			self._init = None
		return copy.deepcopy(self._init)
	
	def copy(self):
		return self.set_from_config(self.get_config(), layer_init=copy.deepcopy(self._init))
	
	def get_config(self):
		return {
			'_id': self.id,
			'use_bn': self.use_bn,
			'activation': self.activation,
			'keep_prob': self.keep_prob,
			'pre_activation': self.pre_activation,
		}
	
	@property
	def variable_list(self):
		"""
		beta: mean scale
		gamma: variance scale
		y = gamma * (x - moving_mean) / sqrt(epsilon + moving_variance) + beta
		"""
		if self.use_bn:
			return {
				'moving_mean': 'BatchNorm/moving_mean:0',
				'moving_variance': 'BatchNorm/moving_variance:0',
				'beta': 'BatchNorm/beta:0',
				'gamma': 'BatchNorm/gamma:0',
			}
		else:
			return {}
	
	@staticmethod
	def set_from_config(layer_config, layer_init):
		raise NotImplementedError
	
	def build(self, _input, net, store_output_op):
		raise NotImplementedError

class Identity(BaseLayer):
	def __init__(self, _id ):
		BaseLayer.__init__(self, _id)

	@property
	def layer_str(self):
		return 'C%d,%d,%d' % (self.filter_num, self.kernel_size, self.strides)
		
	@property
	def variable_list(self):
		var_list = {'kernel': 'kernel:0'}
		var_list.update(super(ConvLayer, self).variable_list)
		return var_list
	
	def get_config(self):
		return {
			'name': 'conv',
			'filter_num': self.filter_num,
			'kernel_size': self.kernel_size,
			'strides': self.strides,
			'ops_array':self.ops_array.tolist(),
			**super(ConvLayer, self).get_config(),
		}
	
	@staticmethod
	def set_from_config(layer_config, layer_init=None):
		conv_layer = ConvLayer(**layer_config)
		conv_layer._init = layer_init
		return conv_layer
	
	def build(self, _input, net: BasicModel, store_output_op=False):
		output = _input
		
		if store_output_op:
			self.output_op = output
		return output



class DenseNetLayer(BaseLayer):
	def __init__(self, _id,k,growth_rate,kernel_size=3,use_bn=True, activation='relu', keep_prob=1.0,
		ready=True, pre_activation=True, **kwargs):
		BaseLayer.__init__(self, _id, use_bn, activation, keep_prob, ready, pre_activation)
		self.k=k
		self.growth_rate=growth_rate
		self.kernel_size=kernel_size
		self.bottleneck=ConvLayer(self.id+"_bottleneck",self.k*self.growth_rate, 1,1,\
							use_bn=True, activation='relu', keep_prob=keep_prob, ready=True, pre_activation=True)
		#self.conv=[]
		#for i in range(14):
		self.conv=(ConvLayer(self.id+'_conv_',self.k,kernel_size ,1,\
							use_bn=True, activation='relu', keep_prob=keep_prob, ready=True, pre_activation=True))
	@property
	def layer_str(self):
		return 'C%d,%d,%d' % (self.filter_num, self.kernel_size, self.strides)
		
	@property
	def variable_list(self):
		var_list = {'kernel': 'kernel:0'}
		var_list.update(super(ConvLayer_blocks, self).variable_list)
		return var_list
	
	def get_config(self):
		return {
			'name': 'DenseNetLayer',
			 'k':self.k,
			 'growth_rate':self.growth_rate,
			'kernel_size':self.kernel_size,
			**super(DenseNetLayer, self).get_config(),
		}

	@staticmethod
	def set_from_config(layer_config, layer_init=None,conv_i=0):

		conv_layer = DenseNetLayer(**layer_config)
		conv_layer._init = layer_init
		
		if layer_init is not None:
			for i in range(len(conv_layer.conv_blocks)):
				for j in range(len(conv_layer.conv_blocks)):
					if conv_layer.ops_array[i][j]>1:
						conv_layer.conv_blocks[i][j]._init=layer_init[i][j]
						
		return conv_layer
		

	def build(self, _input, net: BasicModel, store_output_op=False):
		#8个layer

		
		
		with tf.variable_scope(self._id):
			self._scope = tf.get_variable_scope().name
			
			#print(output.get_shape())
			
			
			
			'''for i in range(input_size):
				d=tf.random_uniform([1])
				c=tf.cond(tf.less(d[0], tf.Variable([0.5],dtype=tf.float32)[0])
								,lambda:_input[i]*0 ,lambda: _input[i])#)
				e=tf.cond(net.is_training,lambda:c,lambda:_input[i])
				output_.append(e)
				output_.append(_input[i])'''
			
			#output=tf.concat(_input,3)
			#
			'''o3=[]
			for i in range(6,14):
				o3.append(self.conv[i].build(output, net, store_output_op))
			i2=[]
			for i in range(0,8,2):
				i2.append(tf.add(o3[i],o3[i+1]))
			o2=[]
			for i in range(2,6):
				o2.append(self.conv[i].build(i2[i-2], net, store_output_op))
			i1=[]
			for i in range(0,4,2):
				i1.append(tf.add(o2[i],o2[i+1]))

			o1=[]
			for i in range(0,2):
				o1.append(self.conv[i].build(i1[i], net, store_output_op))

			output=tf.add(o1[0],o1[1])'''
			


			output=self.bottleneck.build(_input, net, store_output_op)
			output=self.conv.build(output, net, store_output_op)
			#print(2,type(output),type(output_[0]))
			return tf.concat([output,_input],3)
			

		

		
	def set_ops(self,i,j,k,t):
		'''kernel_sizes=[0,0,1,3,5]
		keep_prob=1.0
		if(self.ops_array[i]==0 and k!=0):
			
			if k==1:
				self.conv_blocks[i]=PoolLayer_inner(self._id+'_'+str(i),self.number_of_channel_in_block, 'avg',3, 1)
				
			else: 
				self.conv_blocks[i]=ConvLayer(self._id+"_"+str(i),self.number_of_channel_in_block, kernel_sizes[k],1,\
					use_bn=True, activation='relu', keep_prob=keep_prob, ready=True, pre_activation=True)

				
				

		if(self.ops_array[i]==1 and k!=1):
			if k==0:
				self.conv_blocks[i]=PoolLayer_inner(self._id+'_'+str(i),self.number_of_channel_in_block, 'max',3, 1)
			else: 
				self.conv_blocks[i]=ConvLayer(self._id+"_"+str(i),self.number_of_channel_in_block, kernel_sizes[k],1,\
					use_bn=True, activation='relu', keep_prob=keep_prob, ready=True, pre_activation=True)
				
		if(self.ops_array[i]>1):
			if k==0:
				self.conv_blocks[i]=PoolLayer_inner(self._id+'_'+str(i),self.number_of_channel_in_block, 'max',3, 1)
			elif k==1:
				self.conv_blocks[i]=PoolLayer_inner(self._id+'_'+str(i),self.number_of_channel_in_block, 'avg',3, 1)
			else:
				self.conv_blocks[i]=ConvLayer(self._id+"_"+str(i),self.number_of_channel_in_block, kernel_sizes[k],1,\
					use_bn=True, activation='relu', keep_prob=keep_prob, ready=True, pre_activation=True)
			'''
		self.ops_array[i][j][k]=t

	def renew_init(self, net: BasicModel):
		
		return 
class Pooling(BaseLayer):
	def __init__(self, _id, _type, kernel_size=2, strides=2, use_bn=False, activation=None, keep_prob=1.0,
				 ready=True, pre_activation=True, **kwargs):
		BaseLayer.__init__(self, _id, use_bn, activation, keep_prob, ready, pre_activation)
		
		self._type = _type
		self.kernel_size = kernel_size
		self.strides = strides
	
	@property
	def layer_str(self):
		return 'P%d,%d' % (self.kernel_size, self.strides)
		
	def get_config(self):
		return {
			'name': 'pool',
			'_type': self._type,
			'kernel_size': self.kernel_size,
			'strides': self.strides,
			**super(PoolLayer, self).get_config(),
		}
	
	@staticmethod
	def set_from_config(layer_config, layer_init=None):
		pool_layer = PoolLayer(**layer_config)
		pool_layer._init = layer_init
		return pool_layer
	
	def build(self, _input, net: BasicModel, store_output_op=False):
		output = _input
		if not self.ready:
			return output
		with tf.variable_scope(self._id):
			self._scope = tf.get_variable_scope().name
			param_initializer = self.param_initializer
			if self.pre_activation:
				# batch normalization
				if self.use_bn:
					output = BasicModel.batch_norm(output, net.is_training, net.net_config.bn_epsilon,
												   net.net_config.bn_decay, param_initializer=param_initializer)
				# activation
				output = BasicModel.activation(output, self.activation)
				# Pooling
				if self._type == 'avg':

					output = BasicModel.avg_pool(output, k=self.kernel_size, s=self.strides)
				elif self._type == 'max':
					output = BasicModel.max_pool(output, k=self.kernel_size, s=self.strides)
				else:
					raise ValueError('Do not support the pooling type: %s' % self._type)
			else:
				# Pooling
				if self._type == 'avg':
					output = BasicModel.avg_pool(output, k=self.kernel_size, s=self.strides)
				elif self._type == 'max':
					output = BasicModel.max_pool(output, k=self.kernel_size, s=self.strides)
				else:
					raise ValueError('Do not support the pooling type: %s' % self._type)
				# batch normalization
				if self.use_bn:
					output = BasicModel.batch_norm(output, net.is_training, net.net_config.bn_epsilon,
												   net.net_config.bn_decay, param_initializer=param_initializer)
				# activation
				output = BasicModel.activation(output, self.activation)
			# dropout
			output = BasicModel.dropout(output, self.keep_prob, net.is_training)
		if store_output_op:
			self.output_op = output
		return output

class ConvLayer(BaseLayer):
	def __init__(self, _id, filter_num, kernel_size=3, strides=1,
				 use_bn=True, activation='relu', keep_prob=1.0, ready=True, pre_activation=True, **kwargs):
		BaseLayer.__init__(self, _id, use_bn, activation, keep_prob, ready, pre_activation)
		self.filter_num = filter_num
		self.kernel_size = kernel_size
		self.strides = strides
	
	@property
	def layer_str(self):
		return 'C%d,%d,%d' % (self.filter_num, self.kernel_size, self.strides)
		
	@property
	def variable_list(self):
		var_list = {'kernel': 'kernel:0'}
		var_list.update(super(ConvLayer, self).variable_list)
		return var_list
	
	def get_config(self):
		return {
			'name': 'conv',
			'filter_num': self.filter_num,
			'kernel_size': self.kernel_size,
			'strides': self.strides,
			**super(ConvLayer, self).get_config(),
		}
	
	@staticmethod
	def set_from_config(layer_config, layer_init=None):
		conv_layer = ConvLayer(**layer_config)
		conv_layer._init = layer_init
		return conv_layer
	
	def build(self, _input, net: BasicModel, store_output_op=False):
		output = _input
		if not self.ready:
			return output
		with tf.variable_scope(self._id):
			self._scope = tf.get_variable_scope().name
			output=tf.concat(output,3)
			param_initializer = self.param_initializer
			if self._id=="conv_adjust":
				print("no bn relu")
				output =  BasicModel.conv2d(output, self.filter_num, self.kernel_size, self.strides,
										   param_initializer=param_initializer)
				return output
			if self.pre_activation:
				# batch normalization
				if self.use_bn:
					output = BasicModel.batch_norm(output, net.is_training, net.net_config.bn_epsilon,
												   net.net_config.bn_decay, param_initializer=param_initializer)
				# activation
				output = BasicModel.activation(output, self.activation)
				# convolutional
				output = BasicModel.conv2d(output, self.filter_num, self.kernel_size, self.strides,
										   param_initializer=param_initializer)
			else:
				# convolutional
				output = BasicModel.conv2d(output, self.filter_num, self.kernel_size, self.strides,
										   param_initializer=param_initializer)
				# batch normalization
				if self.use_bn:
					output = BasicModel.batch_norm(output, net.is_training, net.net_config.bn_epsilon,
												   net.net_config.bn_decay, param_initializer=param_initializer)
				# activation
				output = BasicModel.activation(output, self.activation)
			# dropout
			output = BasicModel.dropout(output, self.keep_prob, net.is_training)
		if store_output_op:
			self.output_op = output
		return output
	
	
		
		
class FCLayer(BaseLayer):
	def __init__(self, _id, units, use_bn=False, use_bias=True, activation=None, keep_prob=1.0, ready=True,
				 pre_activation=False, **kwargs):
		BaseLayer.__init__(self, _id, use_bn, activation, keep_prob, ready, pre_activation)
		self.units = units
		self.use_bias = use_bias
	
	@property
	def layer_str(self):
		return 'FC%d' % self.units
	
	@property
	def variable_list(self):
		var_list = {'W': 'W:0'}
		if self.use_bias:
			var_list['bias'] = 'bias:0'
		var_list.update(super(FCLayer, self).variable_list)
		return var_list
	
	def get_config(self):
		return {
			'name': 'fc',
			'units': self.units,
			'use_bias': self.use_bias,
			**super(FCLayer, self).get_config(),
		}
	
	@staticmethod
	def set_from_config(layer_config, layer_init=None):
		fc_layer = FCLayer(**layer_config)
		fc_layer._init = layer_init
		return fc_layer
	
	def build(self, _input, net: BasicModel, store_output_op=False):
		output = _input
		
		with tf.variable_scope(self._id):
			self._scope = tf.get_variable_scope().name
			param_initializer = self.param_initializer
			# flatten if not
			output = BasicModel.flatten(output)
			if self.pre_activation:
				# batch normalization
				if self.use_bn:
					output = BasicModel.batch_norm(output, net.is_training, net.net_config.bn_epsilon,
												   net.net_config.bn_decay, param_initializer=param_initializer)
				# activation
				output = BasicModel.activation(output, self.activation)
				# FC
				output = BasicModel.fc_layer(output, self.units, self.use_bias, param_initializer=param_initializer)
			else:
				# FC
				output = BasicModel.fc_layer(output, self.units, self.use_bias, param_initializer=param_initializer)
				# batch normalization
				if self.use_bn:
					output = BasicModel.batch_norm(output, net.is_training, net.net_config.bn_epsilon,
												   net.net_config.bn_decay, param_initializer=param_initializer)
				# activation
				output = BasicModel.activation(output, self.activation)
			# dropout
			output = BasicModel.dropout(output, self.keep_prob, net.is_training)
		if store_output_op:
			self.output_op = output
		return output


class PoolLayer_blocks(BaseLayer):
	def __init__(self, _id, _type,kernel_size=2,number_of_block=6, strides=2, use_bn=False, activation=None, keep_prob=0.7,
				 ready=True, pre_activation=True, **kwargs):
		BaseLayer.__init__(self, _id, use_bn, activation, keep_prob, ready, pre_activation)
		
		self._type = _type
		self.kernel_size = kernel_size
		self.strides = strides
		self.number_of_block=number_of_block
		pool_blocks=[]
		if keep_prob<1.0:
			print("dropout")
		for i in range(self.number_of_block):
				pool_blocks.append(PoolLayer(_id+'_'+str(i), _type, kernel_size, strides, use_bn, activation, keep_prob,
				 ready, pre_activation))
		self.pool_blocks=pool_blocks
	
	@property
	def layer_str(self):
		return 'P%d,%d' % (self.kernel_size, self.strides)
		
	def get_config(self):
		return {
			'name': 'pool',
			'_type': self._type,
			'kernel_size': self.kernel_size,
			'number_of_block':self.number_of_block,
			
			'strides': self.strides,
			'number_of_block':self.number_of_block,
			**super(PoolLayer_blocks, self).get_config(),
		}
	
	@staticmethod
	def set_from_config(layer_config, layer_init=None):
		pool_layer = PoolLayer_blocks(**layer_config)
		pool_layer._init = layer_init
		return pool_layer
	
	def build(self, _input, net: BasicModel, store_output_op=False):
		output = _input
		if not self.ready:
			return output
		with tf.variable_scope(self._id):
			self._scope = tf.get_variable_scope().name
			param_initializer = self.param_initializer
			output_=[]
			for _i,pool_block in enumerate(self.pool_blocks):
				if output[_i] is None:
					output_.append(None)
				else:
					output_.append(pool_block.build(output[_i],net,store_output_op))
		if store_output_op:
			self.output_op = output_
		return output_


class PoolLayer(BaseLayer):
	def __init__(self, _id, _type, kernel_size=2, strides=2, use_bn=False, activation=None, keep_prob=1.0,
				 ready=True, pre_activation=True, **kwargs):
		BaseLayer.__init__(self, _id, use_bn, activation, keep_prob, ready, pre_activation)
		
		self._type = _type
		self.kernel_size = kernel_size
		self.strides = strides
	
	@property
	def layer_str(self):
		return 'P%d,%d' % (self.kernel_size, self.strides)
		
	def get_config(self):
		return {
			'name': 'pool',
			'_type': self._type,
			'kernel_size': self.kernel_size,
			'strides': self.strides,
			**super(PoolLayer, self).get_config(),
		}
	
	@staticmethod
	def set_from_config(layer_config, layer_init=None):
		pool_layer = PoolLayer(**layer_config)
		pool_layer._init = layer_init
		return pool_layer
	
	def build(self, _input, net: BasicModel, store_output_op=False):
		output =[]
		if isinstance(_input,list):
			_input=tf.concat(_input,3)
		if not self.ready:
			return output
		with tf.variable_scope(self._id):
			self._scope = tf.get_variable_scope().name
			param_initializer = self.param_initializer
			
			if self.pre_activation:
				# batch normalization
				if self.use_bn:
					output = BasicModel.batch_norm(output, net.is_training, net.net_config.bn_epsilon,
												   net.net_config.bn_decay, param_initializer=param_initializer)
				# activation
				output = BasicModel.activation(output, self.activation)
				# FC
				if self._type == 'avg':
				#for i in range(len(_input)):
					output=BasicModel.avg_pool(_input, k=self.kernel_size, s=self.strides)
				elif self._type == 'max':
					output = BasicModel.max_pool(output, k=self.kernel_size, s=self.strides)
				else:
					raise ValueError('Do not support the pooling type: %s' % self._type)
			else:
				if self._type == 'avg':
				#for i in range(len(_input)):
					output.append(BasicModel.avg_pool(_input, k=self.kernel_size, s=self.strides))
				elif self._type == 'max':
					output = BasicModel.max_pool(output, k=self.kernel_size, s=self.strides)
				else:
					raise ValueError('Do not support the pooling type: %s' % self._type)

				if self.use_bn:
					output = BasicModel.batch_norm(output, net.is_training, net.net_config.bn_epsilon,
												   net.net_config.bn_decay, param_initializer=param_initializer)
				# activation
				output = BasicModel.activation(output, self.activation)
					
			# dropout
			output = BasicModel.dropout(output, self.keep_prob, net.is_training)
		if store_output_op:
			self.output_op = output
		return output
		if store_output_op:
			self.output_op = output
		return [output]
class PoolLayer_inner(BaseLayer):
	def __init__(self, _id,filter_num, _type, kernel_size=2, strides=2, use_bn=False, activation=None, keep_prob=1.0,
				 ready=True, pre_activation=True, **kwargs):
		BaseLayer.__init__(self, _id, use_bn, activation, keep_prob, ready, pre_activation)
		self.filter_num = filter_num
		self._type = _type
		self.kernel_size = kernel_size
		self.strides = strides
	
	@property
	def layer_str(self):
		return 'P%d,%d' % (self.kernel_size, self.strides)
		
	def get_config(self):
		return {
			'name': 'pool',
			'_type': self._type,
			'kernel_size': self.kernel_size,
			'strides': self.strides,
			'filter_num':filter_num,
			**super(PoolLayer, self).get_config(),
		}
	
	@staticmethod
	def set_from_config(layer_config, layer_init=None):
		pool_layer = PoolLayer(**layer_config)
		pool_layer._init = layer_init
		return pool_layer
	
	def build(self, _input, net: BasicModel, store_output_op=False):
		output = _input
		if not self.ready:
			return output
		with tf.variable_scope(self._id+"pool"):
			self._scope = tf.get_variable_scope().name
			param_initializer = self.param_initializer
			output=tf.concat(output,3)
			output = BasicModel.batch_norm(output, net.is_training, net.net_config.bn_epsilon,
												   net.net_config.bn_decay, param_initializer=param_initializer)
			output = BasicModel.activation(output, 'relu')
			output = BasicModel.conv2d(output, self.filter_num, 1, 1,
										   param_initializer=param_initializer)
			if self._type == 'avg':
				output = BasicModel.avg_pool(output, k=self.kernel_size, s=self.strides)
			elif self._type == 'max':
					output = BasicModel.max_pool(output, k=self.kernel_size, s=self.strides)
			else:
					raise ValueError('Do not support the pooling type: %s' % self._type)
			
			

			
			output = BasicModel.dropout(output, self.keep_prob, net.is_training)
		

		if store_output_op:
			self.output_op = output
		return [output]


class ConvLayer_blocks(BaseLayer):
	def __init__(self, _id,k,number_of_input,kernel_size=3,strides=1,use_bn=True, activation='relu', keep_prob=1.0,
		ready=True, pre_activation=True, **kwargs):
		BaseLayer.__init__(self, _id, use_bn, activation, keep_prob, ready, pre_activation)
		self.k=k
		self.number_of_input=number_of_input

		convs=[]
		for i in range(number_of_input):
			convs.append(ConvLayer(self.id+"_"+str(i),self.k,kernel_size,strides,\
					use_bn=True, activation='relu', keep_prob=keep_prob, ready=True, pre_activation=True))

		
		self.convs=convs
	
	@property
	def layer_str(self):
		return 'C%d,%d,%d' % (self.filter_num, self.kernel_size, self.strides)
		
	@property
	def variable_list(self):
		var_list = {'kernel': 'kernel:0'}
		var_list.update(super(ConvLayer_blocks, self).variable_list)
		return var_list
	
	def get_config(self):
		return {
			'name': 'conv_block',
			 'number_of_channel_in_block':self.number_of_channel_in_block,
			'ops_array':self.ops_array,
			"number_of_block":self.number_of_block,
			**super(ConvLayer_blocks, self).get_config(),
		}
	
	@staticmethod
	def set_from_config(layer_config, layer_init=None,conv_i=0):

		conv_layer = ConvLayer_blocks(**layer_config)
		conv_layer._init = layer_init
		
		if layer_init is not None:
			for i in range(len(conv_layer.conv_blocks)):
				for j in range(len(conv_layer.conv_blocks)):
					if conv_layer.ops_array[i][j]>1:
						conv_layer.conv_blocks[i][j]._init=layer_init[i][j]
						
		return conv_layer
	
	def build(self, _input, net: BasicModel, store_output_op=False):


		if not self.ready:
			return output
		
		with tf.variable_scope(self._id):
			self._scope = tf.get_variable_scope().name
			output_=[]
			for _i,conv_i in enumerate(self.convs):
				output_.append(conv_i.build(_input[_i], net, store_output_op))
			
		
		if store_output_op:
			self.output_op = output_
		
		return output_
	def set_ops(self,i,j,k):
		kernel_sizes=[0,0,1,3,5,7]

		if(self.ops_array[i][j]==0 and k!=0):
			
			if k==1:
				self.conv_blocks[i][j]=Identity(self.id+"_"+str(i)+"_"+str(j),self.is_first_layer)
			else: 
				self.conv_blocks[i][j]=ConvLayer(self.id+"_"+str(i)+"_"+str(j),self.number_of_channel_in_block, kernel_sizes[k],1,\
					use_bn=True, activation='relu', keep_prob=1.0, ready=True, pre_activation=True)
				

		if(self.ops_array[i][j]==1 and k!=1):
			if k==0:
				self.conv_blocks[i][j]=0
			else: 
				self.conv_blocks[i][j]=ConvLayer(self.id+"_"+str(i)+"_"+str(j),self.number_of_channel_in_block, kernel_sizes[k],1,\
					use_bn=True, activation='relu', keep_prob=1.0, ready=True, pre_activation=True)
				
		if(self.ops_array[i][j]>1):
			if k==0:
				self.conv_blocks[i][j]=0
			elif k==1:
				self.conv_blocks[i][j]=Identity(self.id+"_"+str(i)+"_"+str(j),self.is_first_layer)
			else:
				self.conv_blocks[i][j]=ConvLayer(self.id+"_"+str(i)+"_"+str(j),self.number_of_channel_in_block, kernel_sizes[k],1,\
					use_bn=True, activation='relu', keep_prob=1.0, ready=True, pre_activation=True)
			
		self.ops_array[i][j]=k

	def renew_init(self, net: BasicModel):
		_init=[]
		for i in range(len(self.conv_blocks)):
			row=[]
			for j in range(len(self.conv_blocks)) :
				if(self.ops_array[i][j]>1):
					row.append(self.conv_blocks[i][j].renew_init(net))
				else:
					row.append(0)
			_init.append(row)

		self._init=_init
		return copy.deepcopy(self._init)


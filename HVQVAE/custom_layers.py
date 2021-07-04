import tensorflow as tf
import numpy as np

class VectorQuantizer(tf.keras.layers.Layer):  

    """ Implements the Vector Quantizer, the layer returns directly the quantized indices z_q"""
    
    def __init__(self, k, **kwargs):
        super(VectorQuantizer, self).__init__(**kwargs)
        self.k = k
    
    #Perhaps add a method that gives the codebook directly from here
    
    def build(self, input_shape):
        self.d = int(input_shape[-1])
        rand_init = tf.keras.initializers.VarianceScaling(distribution="uniform")
        self.codebook = self.add_weight(name='w',shape=(self.k, self.d), initializer=rand_init, trainable=True)
        
    def call(self, inputs):
        # Map z_e of shape (b, w,, h, d) to indices in the codebook
        self.lookup_ = tf.reshape(self.codebook, shape=(1, 1, 1, self.k, self.d))
        z_e = tf.expand_dims(inputs, -2)
        dist = tf.norm(z_e - self.lookup_, axis=-1)
        k_index = tf.argmin(dist, axis=-1)
        k_index_one_hot = tf.one_hot(k_index, self.k)
        z_q = self.lookup_ * k_index_one_hot[..., None]
        z_q = tf.reduce_sum(z_q, axis=-2)
        return z_q
    

class CBAM(tf.keras.layers.Layer):

    """Implements Covolutional Bloack Attention Module as arXiv:1807.06521"""

    def __init__(self, filter_size, activation=tf.keras.activations.relu, dilation=1, kernel_size=3, bottleneck_reduction=4, renorm=False):
        super(CBAM, self).__init__()
        self.reduction=bottleneck_reduction
        self.renorm=renorm
        self.filter_size=filter_size
        self.activation=activation
        self.MLP = self.build_MLP()
        self.conv=tf.keras.layers.Conv2D(filters=filter_size, kernel_size=kernel_size,padding='same',
                                         dilation_rate=dilation, activation='sigmoid')

    def build_MLP(self):
        inputs=tf.keras.layers.Input(shape=self.filter_size)
        dense=tf.keras.layers.Dense(self.filter_size//self.reduction, activation=self.activation)(inputs)
        dense=tf.keras.layers.Dense(self.filter_size, activation=self.activation)(dense)
        dense=tf.keras.layers.BatchNormalization(renorm=self.renorm)(dense) 
        dense=tf.keras.layers.Reshape((1,1,self.filter_size))(dense)

        MLP=tf.keras.Model(inputs=inputs, outputs=dense)

        return MLP

    def call(self, inputs):
        avg=tf.keras.layers.GlobalAveragePooling2D()(inputs)
        maxi=tf.keras.layers.GlobalMaxPooling2D()(inputs)
        avg=self.MLP(avg)
        maxi=self.MLP(maxi)
        channel_attention=tf.keras.activations.sigmoid(avg+maxi)
        channel_conditioned=tf.math.multiply(inputs, channel_attention)
        channel_average=tf.keras.backend.mean(channel_conditioned, axis=-1,  keepdims=True)
        channel_max= tf.keras.backend.max(channel_conditioned, axis=-1,  keepdims=True)
        spatial_reduction=tf.keras.layers.Concatenate(axis=-1)([channel_average, channel_max])
        spatial_attention=self.conv(spatial_reduction)
        spatial_conditioned=tf.math.multiply(channel_conditioned, spatial_attention )      
        
        return spatial_conditioned
       
class GateActivation(tf.keras.layers.Layer):

    """Implements gated activation for PixelCNN"""
    
    def __init__(self, **kwargs):
        super(GateActivation, self).__init__()
    def call(self, inputs):
        x, y = tf.split(inputs, 2, axis=-1)
        return tf.math.tanh(x) * tf.math.sigmoid(y)
        
class CausalAverage(tf.keras.layers.Layer):

    """Given a 2D input returns a 2D tensor with the same shape in which each entry is 
    the causal average of the previous entries, i.e. it is the sum of all the entries on the top and left 
    divided by the number of such entries"""
    
    def __init__(self, **kwargs):
        super(CausalAverage, self).__init__()
        
    def build(self, input_shape):   
        self.b=input_shape[0]
        self.h=input_shape[1]
        self.w=input_shape[2]
        self.c=input_shape[3]
        M=np.ones((self.h, self.w))
        M=tf.convert_to_tensor(M,dtype='float32')
        M=tf.reshape(M,(self.h*self.w,))
        M=tf.cumsum(M)
        M=tf.math.reciprocal(M)
        self.Normalizing_matrix=tf.reshape(M,[1,self.h*self.w,1]) #Normalizing matrix just used to turn cumulative sum into cumulative average

    def call(self, inputs): 
        reshape=tf.keras.layers.Reshape((self.h*self.w,self.c))(inputs) #Flattens the input
        cum_sum=tf.cumsum(reshape, axis=1) #Cumulative sum over the flatten input will sum over the desired range
        normalize=tf.math.multiply(cum_sum, self.Normalizing_matrix)       
        average=tf.keras.layers.Reshape((self.h,self.w,self.c))(normalize)
        
        return average
        
        
class CausalAttentionModule(tf.keras.layers.Layer):

    """Implement a causal version of the CBAM module"""
    
    def __init__(self, filter_size,mask_type, activation=tf.keras.activations.relu, dilation=1, bottleneck_reduction=4):
        super(CausalAttentionModule, self).__init__()
        self.filter_size=filter_size
        self.activation=activation
        self.mask_type=mask_type
        self.reduction=bottleneck_reduction

      
    def build(self,input_shape):
        self.masked_conv=MaskedConv2D(self.mask_type,filters=self.filter_size, kernel_size=5,padding='same', activation='sigmoid')
        self.channel_conv1=tf.keras.layers.Conv2D(filters=self.filter_size//self.reduction, kernel_size=1,activation=self.activation)
        self.channel_conv2=tf.keras.layers.Conv2D(filters=self.filter_size, kernel_size=1, activation=self.activation)
        self.bn=tf.keras.layers.BatchNormalization()
        self.bn1=tf.keras.layers.BatchNormalization()
        self.causal_average=CausalAverage()
        
    def call(self, inputs):
        avg=self.causal_average(inputs)
        avg=self.channel_conv1(avg)
        avg=self.channel_conv2(avg)
        avg=self.bn(avg)
        channel_attention=tf.keras.activations.sigmoid(avg)
        channel_conditioned=tf.math.multiply(inputs, channel_attention)
        channel_average=tf.keras.backend.mean(channel_conditioned, axis=-1,  keepdims=True)
        channel_max= tf.keras.backend.max(channel_conditioned, axis=-1,  keepdims=True)
        spatial_reduction=tf.keras.layers.Concatenate(axis=-1)([channel_average, channel_max])
        spatial_reduction=self.bn1(spatial_reduction)
        spatial_attention=self.masked_conv(spatial_reduction) 
        spatial_conditioned=tf.math.multiply(channel_conditioned, spatial_attention )       
        return spatial_conditioned

class MaskedConv2D(tf.keras.layers.Layer):

    """Masked convolutional layer"""
    
    def __init__(self, mask_type, **kwargs):
        super(MaskedConv2D, self).__init__()
        self.mask_type = mask_type
        self.conv = tf.keras.layers.Conv2D(**kwargs)

    def build(self, input_shape):
        # Build the conv2d layer to initialize kernel variables
        self.conv.build(input_shape)
        # Use the initialized kernel to create the mask
        kernel_shape = self.conv.kernel.get_shape()
        self.mask = np.zeros(shape=kernel_shape)
        self.mask[: kernel_shape[0] // 2, ...] = 1.0
        self.mask[kernel_shape[0] // 2, : kernel_shape[1] // 2, ...] = 1.0
        if self.mask_type == 'b': #Type of mask, 'b' includes the current pixel, 'a' doesn't. 'a' is used only for the first layer
            self.mask[kernel_shape[0] // 2, kernel_shape[1] // 2, ...] = 1.0
        #print below is useful to check the kernel
        #print("Mask ", self.mask[:,:,0,0])

    def call(self, inputs):
        self.conv.kernel.assign(self.conv.kernel * self.mask)
        return self.conv(inputs)
    


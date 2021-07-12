"""This module contains the architecture building blocks of the hierarchical VQ-VAE, including
encoders, quantizers and decoders"""

import tensorflow as tf
from HVQVAE.custom_layers import VectorQuantizer, CBAM
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, Concatenate, BatchNormalization, Lambda
from tensorflow.keras import Model, Input
from HVQVAE.hyperparameters import KT,DT,KM,DM, KB,DB, img_size, color_channels

#Weights for the commitment loss
top_beta=0.25
mid_beta=0.25
bot_beta=0.25             
Tencoder_layers= [128,128,128,128,128]      #Strided layers for the top-level encoder    
Mencoder_layers=[128,128,128,128]       #Strided layers for the mid-level encoder
Bencoder_layers=[128,128,128]           #Strided layers for the bottom-level encoder
Bdecoder_layers=[128,128,128]           #Strided layers for the decoder
image_shape=[img_size,img_size,color_channels]
T_reduction=2**len(Tencoder_layers)
T_dim=[image_shape[0]//T_reduction, image_shape[1]//T_reduction]
M_reduction=2**len(Mencoder_layers)
M_dim=[image_shape[0]//M_reduction, image_shape[1]//M_reduction]
B_reduction=2**len(Bencoder_layers)
B_dim=[image_shape[0]//B_reduction, image_shape[1]//B_reduction]
ACT=tf.keras.layers.ELU(alpha=0.1)

def res_block(inputs, filters, attention_dilation=1, attention_kernel=3):
    """Residual block to be used in various pieces of the architecture"""
    x=Conv2D(filters=filters, kernel_size=1, padding='same', activation=ACT)(inputs) 
    skip=x
    x=Conv2D(filters=filters//2, kernel_size=3, padding='same', activation=ACT)(x) 
    attention=CBAM(filters//2, activation=ACT,dilation=attention_dilation, kernel_size=attention_kernel)(x)
    x=Conv2D(filters=filters, kernel_size=1, padding='same', activation=ACT)(x+attention) 
    out=BatchNormalization()(x+skip)
    return out
    
def build_top_encoder(input_shape, d=DT, layers=Tencoder_layers):
    """Return the encoder for top level latents""" 
    encoder_inputs = Input(shape=input_shape, name='encoder_inputs')
    x=encoder_inputs
    for i, filters in enumerate(layers):
        x = Conv2D(filters=filters, kernel_size=3, padding='SAME', activation=ACT, 
                            strides=2, name="EncoderConv{}".format(i + 1))(x)
        x=res_block(x,filters=filters, attention_dilation=3, attention_kernel=5)

    z_e = Conv2D(filters=d, kernel_size=1, padding='SAME', strides=(1, 1), name='z_e')(x)
    encoder=Model(inputs=encoder_inputs, outputs=z_e, name='Top_encoder')
    return encoder

def build_mid_encoder(top_input_shape,mid_input_shape, d=DM, layers=Mencoder_layers):
    """Return the encoder for middle level latents""" 
    top_inputs = Input(shape=top_input_shape, name='top_encoder_inputs')
    mid_inputs = Input(shape=mid_input_shape, name='mid_encoder_inputs')
    top=top_inputs
    x=mid_inputs
    for i, filters in enumerate(layers):
          x = Conv2D(filters=filters, kernel_size=3, padding='same', activation=ACT, 
                            strides=2, name="EncoderConv{}".format(i + 1))(x)
          x=res_block(x,filters, attention_dilation=5, attention_kernel=5)
    for i in range(len(Tencoder_layers)-len(Mencoder_layers)):
        top=Conv2DTranspose(filters=DM, kernel_size=4, strides=2, padding='same')(top)
    x=Concatenate(axis=-1)([top, x])
    x=BatchNormalization()(x)
    x=res_block(x,256, attention_dilation=3, attention_kernel=5)
    x=res_block(x,256, attention_dilation=3, attention_kernel=5)
    z_e = Conv2D(filters=d, kernel_size=1, padding='same', name='z_e')(x)
    encoder=Model(inputs=[top_inputs, mid_inputs], outputs=z_e, name='Mid_encoder')
    return encoder

def build_bot_encoder(top_input_shape,mid_input_shape,bot_input_shape, d=DB, layers=Bencoder_layers):
    """Return the encoder for bottom level latents""" 
    top_inputs = Input(shape=top_input_shape, name='top_encoder_inputs')
    mid_inputs = Input(shape=mid_input_shape, name='mid_encoder_inputs')
    bot_inputs = Input(shape=bot_input_shape, name='bot_encoder_inputs')
    top_to_bot=top_inputs
    for i in range(len(Tencoder_layers)-len(Bencoder_layers)):
       top_to_bot=Conv2DTranspose(filters=DB, kernel_size=4, strides=2, padding='same',activation=ACT)(top_to_bot) 
    mid_to_bot=mid_inputs
    for i in range(len(Mencoder_layers)-len(Bencoder_layers)):
       mid_to_bot=Conv2DTranspose(filters=DB, kernel_size=4, strides=2, padding='same',activation=ACT)(mid_to_bot) 
    x=bot_inputs
    for i, filters in enumerate(layers):
        x= Conv2D(filters=filters, kernel_size=3, padding='SAME', activation=ACT, 
                            strides=2, name="EncoderConv{}".format(i + 1))(x)
        x=res_block(x,filters, attention_dilation=5, attention_kernel=5) #new addition
    x=Concatenate(axis=-1)([x,top_to_bot, mid_to_bot])
    x=BatchNormalization()(x)
    x=res_block(x,256, attention_dilation=5, attention_kernel=5)
    x=res_block(x,256, attention_dilation=5, attention_kernel=5)
    x=res_block(x,256, attention_dilation=5, attention_kernel=5)
    z_e = Conv2D(filters=d, kernel_size=1, padding='SAME',  name='Bot_encoder')(x)
    encoder=Model(inputs=[top_inputs, mid_inputs,bot_inputs], outputs=z_e, name='Bot_encoder')
    return encoder
   
def build_quantizer(input_shape,d, k, beta=0.25, level='quantizer'):
    """Return the quantizer for the specified level"""
    dim1=input_shape[0]
    dim2=input_shape[1]
    quantizer_input=Input([dim1, dim2, d], name='{}_quantizer_inputs'.format(level))
    z_e=quantizer_input
    z_q=VectorQuantizer(k, name="Vector_Quantizer".format(level))(z_e)
    #straight through estimator for gradients
    straight_through =Lambda(lambda x : x[1] + tf.stop_gradient(x[0] - x[1]),
                             name="{}_straight_through_estimator".format(level))([z_q,z_e])
    vq_loss = tf.reduce_mean((tf.stop_gradient(z_e) - z_q)**2)
    commit_loss = tf.reduce_mean((z_e - tf.stop_gradient(z_q))**2)
    latent_loss = tf.identity(vq_loss +beta * commit_loss, name="{}_latent_loss".format(level))
    quantizer=Model(inputs=quantizer_input, outputs=straight_through)
    quantizer.add_loss(latent_loss)
    quantizer.add_metric(latent_loss, name="{}_latent_codes_loss".format(level), aggregation='mean')
    return quantizer
    
#The decoder takes as inputs both the entry in the codebook and the compressed image   
def build_decoder(T_shape,M_shape, B_shape, layers=[32,32]):
    """Return the decoder"""
    top_inputs=Input(shape=T_shape, name='decoder_top_inputs')
    middle_inputs=Input(shape=M_shape, name='decoder_mid_inputs')
    bottom_inputs=Input(shape=B_shape, name='decoder_bottom_inputs')
    top_upsample=Conv2D(filters=256, kernel_size=1,padding="SAME", activation=ACT)(top_inputs)
    for i in range(len(Tencoder_layers)):
           top_upsample=Conv2DTranspose(filters=128, kernel_size=4, padding="SAME", activation=ACT,strides=2)(top_upsample)
           top_upsample=BatchNormalization()(top_upsample)
    mid_upsample=Conv2D(filters=128, kernel_size=1,padding="SAME", activation=ACT)(middle_inputs)
    for i in range(len(Mencoder_layers)):
           mid_upsample=Conv2DTranspose(filters=128, kernel_size=4, padding="SAME", activation=ACT,strides=2)(mid_upsample)
           mid_upsample=BatchNormalization()(mid_upsample)
    bot_upsample=Conv2D(filters=128, kernel_size=1,padding="SAME", activation=ACT)(bottom_inputs)
    for i, filters in enumerate(layers):
           bot_upsample = Conv2DTranspose(filters=filters, kernel_size=4, strides=2, padding="same", 
                                          activation=ACT, name="convT{}".format(i + 1))(bot_upsample)
           bot_upsample=BatchNormalization()(bot_upsample)
    #Combine together the three upsampled latents by adding them
    y=tf.keras.layers.Add()([bot_upsample, mid_upsample, top_upsample]) 
    y=Conv2D(filters=256, kernel_size=1, activation=ACT)(y)
    y=Conv2D(filters=256, kernel_size=1, activation=ACT)(y)
    y=Conv2D(filters=256, kernel_size=1, activation=ACT)(y)
    y=BatchNormalization()(y)
    reconstructed=Conv2D(filters=3, kernel_size=1,padding="same", activation=ACT, name='output')(y)
    decoder=Model(inputs=[top_inputs,middle_inputs,bottom_inputs], outputs=reconstructed, name='decoder')
    return decoder
 
def build_VQVAE(top_encoder, top_quantizer, mid_encoder, mid_quantizer, bottom_encoder, bottom_quantizer,decoder):   
    """Assemble encoders, quantizers and decoder into the entire VQVAE architecture"""  
    input_shape=image_shape
    vqvae_input=tf.keras.Input(input_shape, name='Input')
    top_encoded=top_encoder(vqvae_input)
    top_quantized=top_quantizer(top_encoded)
    mid_encoded=mid_encoder([top_quantized,vqvae_input])
    mid_quantized=mid_quantizer(mid_encoded)
    bottom_encoded=bottom_encoder([top_quantized, mid_quantized, vqvae_input])
    bottom_quantized=bottom_quantizer(bottom_encoded)
    bottom_decoded=decoder([top_quantized, mid_quantized, bottom_quantized])
    vqvae=Model(inputs=vqvae_input, outputs=bottom_decoded, name='VQVAE')
    return vqvae



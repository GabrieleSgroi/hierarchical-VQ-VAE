"""This module contains the architecture building blocks of the hierarchical VQ-VAE, including
encoders, quantizers and decoders"""

import tensorflow as tf
import os
import HVQVAE.hyperparameters as hyper
from HVQVAE.custom_layers import VectorQuantizer, CBAM
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, Concatenate, BatchNormalization, Lambda
from tensorflow.keras import Model

current_path = os.path.dirname(os.path.realpath(__file__))

KT=hyper.KT               
DT=hyper.DT                
KM=hyper.KM                
DM=hyper.DM                
KB=hyper.KB               
DB=hyper.DB

ACT=tf.keras.layers.ELU(alpha=0.1)


# Weights for the commitment loss
top_beta=0.25
mid_beta=0.25
bot_beta=0.25             

Tencoder_layers= [128,128,128,128,128]      #Strided layers for the top-level encoder    
Mencoder_layers=[128,128,128,128]       #Strided layers for the mid-level encoder
Bencoder_layers=[128,128,128]           #Strided layers for the bottom-level encoder
Bdecoder_layers=[256,256,256]           #Strided layers for the decoder
img_size=512
color_channels=3
image_shape=[img_size,img_size,color_channels]
T_reduction=2**len(Tencoder_layers)
T_dim=[image_shape[0]//T_reduction, image_shape[1]//T_reduction]
M_reduction=2**len(Mencoder_layers)
M_dim=[image_shape[0]//M_reduction, image_shape[1]//M_reduction]
B_reduction=2**len(Bencoder_layers)
B_dim=[image_shape[0]//B_reduction, image_shape[1]//B_reduction]

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
    encoder_inputs = tf.keras.Input(shape=input_shape, name='encoder_inputs')
    x=encoder_inputs
    for i, filters in enumerate(layers):
        x = Conv2D(filters=filters, kernel_size=3, padding='SAME', activation=ACT, 
                            strides=2, name="EncoderConv{}".format(i + 1))(x)
        x=res_block(x,filters=filters, attention_dilation=3, attention_kernel=5)

    x=res_block(x,filters=256, attention_dilation=2, attention_kernel=5)
 
    z_e = Conv2D(filters=d, kernel_size=1, padding='SAME', strides=(1, 1), name='z_e')(x)
    
    encoder=Model(inputs=encoder_inputs, outputs=z_e, name='Top_encoder')
    return encoder

def build_mid_encoder(top_input_shape,mid_input_shape, d=DM, layers=Mencoder_layers):
    top_inputs = tf.keras.Input(shape=top_input_shape, name='top_encoder_inputs')
    mid_inputs = tf.keras.Input(shape=mid_input_shape, name='mid_encoder_inputs')
    top=top_inputs
    x=mid_inputs
    for i, filters in enumerate(layers):
          x = Conv2D(filters=filters, kernel_size=3, padding='same', activation=ACT, 
                            strides=2, name="EncoderConv{}".format(i + 1))(x)
          x=res_block(x,filters, attention_dilation=5, attention_kernel=5)

    for i in range(len(Tencoder_layers)-len(Mencoder_layers)):
        top=Conv2DTranspose(filters=128, kernel_size=4, strides=2, padding='same')(top) #last change filters 16_>DM, take out if it takes too much time
    
    x=Concatenate(axis=-1)([top, x])
    x=res_block(x,256, attention_dilation=5, attention_kernel=5)
    z_e = Conv2D(filters=d, kernel_size=1, padding='same', name='z_e')(x)
    
    encoder=Model(inputs=[top_inputs, mid_inputs], outputs=z_e, name='Mid_encoder')
    return encoder

def build_bot_encoder(top_input_shape,mid_input_shape,bot_input_shape, d=DB, layers=Bencoder_layers):
    top_inputs = tf.keras.Input(shape=top_input_shape, name='top_encoder_inputs')
    mid_inputs = tf.keras.Input(shape=mid_input_shape, name='mid_encoder_inputs')
    bot_inputs = tf.keras.Input(shape=bot_input_shape, name='bot_encoder_inputs')
    top_to_bot=top_inputs
    for i in range(len(Tencoder_layers)-len(Bencoder_layers)):
       top_to_bot=Conv2DTranspose(filters=64, kernel_size=4, strides=2, padding='same',activation=ACT)(top_to_bot) 
    mid_to_bot=mid_inputs
    for i in range(len(Mencoder_layers)-len(Bencoder_layers)):
       mid_to_bot=Conv2DTranspose(filters=64, kernel_size=4, strides=2, padding='same',activation=ACT)(mid_to_bot) 

    x=bot_inputs
    for i, filters in enumerate(layers):
        x= Conv2D(filters=filters, kernel_size=3, padding='SAME', activation=ACT, 
                            strides=2, name="EncoderConv{}".format(i + 1))(x)
        x=res_block(x,filters, attention_dilation=5, attention_kernel=5) #new addition
                    

    x=Concatenate(axis=-1)([x,top_to_bot, mid_to_bot])
    x=BatchNormalization()(x)
    x=res_block(x,256, attention_dilation=5, attention_kernel=5)
    x=res_block(x,256, attention_dilation=5, attention_kernel=5)
    z_e = Conv2D(filters=d, kernel_size=1, padding='SAME', strides=(1, 1), name='z_e')(x)
    
    encoder=Model(inputs=[top_inputs, mid_inputs,bot_inputs], outputs=z_e, name='Bot_encoder')
    return encoder
   
def build_quantizer(input_shape,d, k, beta=0.25, level='quantizer'):
    dim1=input_shape[0]
    dim2=input_shape[1]
    quantizer_input=tf.keras.Input([dim1, dim2, d], name='{}_quantizer_inputs'.format(level))
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
    

def build_decoder(T_shape,M_shape, B_shape, layers=[32,32]):
    Top_inputs=tf.keras.Input(shape=T_shape, name='decoder_top_inputs')
    Middle_inputs=tf.keras.Input(shape=M_shape, name='decoder_mid_inputs')
    Bottom_inputs=tf.keras.Input(shape=B_shape, name='decoder_bottom_inputs')
    
    top_latent=BatchNormalization()(Top_inputs)
    top_to_mid=res_block(top_latent,filters=256, attention_dilation=5, attention_kernel=5)
    for i in range(len(Tencoder_layers)-len(Mencoder_layers)):
           top_to_mid=Conv2DTranspose(filters=DM, kernel_size=4, padding="SAME", activation=ACT,strides=2)(top_to_mid)

    top_to_bot=top_to_mid
    for i in range(len(Tencoder_layers)-len(Mencoder_layers)):
           top_to_bot=Conv2DTranspose(filters=DM, kernel_size=4, padding="SAME", activation=ACT,strides=2)(top_to_bot)

    mid_latent=Concatenate(axis=-1)([top_to_mid, Middle_inputs])
    mid_latent=BatchNormalization()(mid_latent)
    #mid_latent=res_block(mid_latent,filters=256, attention_dilation=5, attention_kernel=5)
    mid_to_bot=res_block(mid_latent,filters=256, attention_dilation=5, attention_kernel=5)
    for i in range(len(Mencoder_layers)-len(Bencoder_layers)):
            mid_to_bot=Conv2DTranspose(filters=DB, kernel_size=4, padding="SAME", 
                                       activation=ACT,strides=(2,2))(mid_to_bot)
    

    bot_latent=Concatenate(axis=-1)([Bottom_inputs, mid_to_bot,top_to_bot])
    bot_latent=BatchNormalization()(bot_latent)
    bot_latent=res_block(bot_latent,filters=256, attention_dilation=5, attention_kernel=5)
    bot_latent=res_block(bot_latent,filters=256, attention_dilation=5, attention_kernel=5)
    #bot_latent=res_block(bot_latent,filters=256, attention_dilation=5, attention_kernel=5)

    for i, filters in enumerate(layers):
           bot_latent = Conv2DTranspose(filters=filters, kernel_size=4, strides=2, padding="same", 
                                        activation=ACT, name="convT{}".format(i + 1))(bot_latent)
    y=BatchNormalization()(bot_latent)
    y=Conv2D(filters=128, kernel_size=1,padding="SAME", activation=ACT)(y)
    reconstructed =Conv2D(filters=3, kernel_size=1,padding="same", activation=ACT, name='output')(y)
    decoder=Model(inputs=[Top_inputs,Middle_inputs,Bottom_inputs], outputs=reconstructed, name='decoder')
    return decoder
 
 
def build_VQVAE(top_encoder, top_quantizer, mid_encoder, mid_quantizer, bottom_encoder, bottom_quantizer,decoder):    
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



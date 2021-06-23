import tensorflow as tf
import os
from HVQVAE.architecture import build_top_encoder, build_mid_encoder, build_bot_encoder, build_quantizer, build_decoder, build_VQVAE


from HVQVAE.hyperparameters import KT,DT,KM,DM, KB,DB

#Weights for the commitment loss
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

def load_top_encoder():
    encoder=build_top_encoder(image_shape,DT,Tencoder_layers)
    url='https://github.com/GabrieleSgroi/hierarchical-VQ-VAE/blob/main/HVQVAE/hvqvae_weights/top_encoder.h5?raw=true'
    weights_dir=tf.keras.utils.get_file('top_encoder_weights',url)
    encoder.load_weights(weights_dir)    
    print("Top encoder loaded")
    return encoder

def load_mid_encoder():
    encoder=build_mid_encoder([image_shape[0]//T_reduction,image_shape[1]//T_reduction,DT],image_shape,DM,Mencoder_layers)
    weights_dir=current_path+"/hvqvae_weights/middle_encoder.h5"
    encoder.load_weights(weights_dir)    
    print("Mid encoder loaded")
    return encoder

def load_bot_encoder():
    encoder=build_bot_encoder([image_shape[0]//T_reduction,image_shape[1]//T_reduction,DT],[image_shape[0]//M_reduction,image_shape[1]//M_reduction,DM],image_shape, DB, Bencoder_layers)
    weights_dir=current_path+"/hvqvae_weights/bottom_encoder.h5"
    encoder.load_weights(weights_dir)    
    print("Bot encoder loaded")
    return encoder

def load_top_quantizer():
    quantizer=build_quantizer(T_dim,DT,KT, top_beta,level='top')
    weights_dir=current_path+"/hvqvae_weights/top_quantizer.h5"
    quantizer.load_weights(weights_dir)
    print("Top quantizer Loaded")
    return quantizer

def load_mid_quantizer():
    quantizer=build_quantizer(M_dim,DM,KM, mid_beta,level='mid')    
    weights_dir=current_path+"/hvqvae_weights/middle_quantizer.h5"
    quantizer.load_weights(weights_dir)
    print("Mid quantizer Loaded")
    return quantizer


def load_bot_quantizer():
    quantizer=build_quantizer(B_dim,DB,KB, bot_beta, level='bot')
    weights_dir=current_path+"/hvqvae_weights/bottom_quantizer.h5"
    quantizer.load_weights(weights_dir)
    print("Bot quantizer Loaded")
    return quantizer

def load_decoder():
    decoder=build_decoder([T_dim[0],T_dim[1],DT],[M_dim[0],M_dim[1],DM],[B_dim[0],B_dim[1],DB], Bdecoder_layers)
    weights_dir=current_path+"/hvqvae_weights/decoder.h5"
    decoder.load_weights(weights_dir)
    print("Decoder loaded")
    return decoder

def load_VQVAE():
    top_encoder=load_top_encoder()
    top_quantizer=load_top_quantizer()
    mid_encoder=load_mid_encoder()
    mid_quantizer=load_mid_quantizer()
    bot_encoder=load_bot_encoder()
    bot_quantizer=load_bot_quantizer()
    decoder=load_decoder()
    vqvae=build_VQVAE(top_encoder, top_quantizer, mid_encoder, mid_quantizer, bot_encoder, bot_quantizer,decoder)
    print("Loaded Hierarchical VQVAE")
    return vqvae








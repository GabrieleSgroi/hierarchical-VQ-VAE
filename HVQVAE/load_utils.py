import tf
from HVQVAE.architecture import build_top_encoder, build_mid_encoer, build_bot_encoder, build_quantizer, build_decoder, build_VQVAE

def load_top_encoder():
    encoder=build_top_encoder(image_shape,DT,Tencoder_layers)
    weights_dir=current_path+"/hvqvae_weights/top_encoder.h5"
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
    weights_dir=current_path+"/hvqvae_weights/bot_encoder.h5"
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
    weights_dir=current_path+"/hvqvae_weights/bot_quantizer.h5"
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
    vqvae=build_VQVAE(top_encoder, top_quantizer, mid_encoder, mid_quantizer, bottom_encoder, bottom_quantizer,decoder)
    print("Loaded Hierarchical VQVAE")
    return vqvae

def get_codebook(quantizer):
    
    """Given a quantizer returns the learned codebook"""
    
    codebook=quantizer.get_weights()[0]
    return codebook

def codebook_from_index(codebook, k_index):
    
    """"Transform indices into the corresponding codebook entries"""
    
    lookup_ = tf.reshape(codebook, shape=(1, 1, 1,KT, DT))
    k_index_one_hot = tf.one_hot(k_index,KT)
    z_q = lookup_ * k_index_one_hot[..., None]
    z_q = tf.reduce_sum(z_q, axis=-2)
    return z_q
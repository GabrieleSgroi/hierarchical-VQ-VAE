from HVQVAE.hyperparameters import KT,DT,KM,DM,KB,DB, img_size, color_channels
import HVQVAE.load_utils as loader
import tensorflow as tf

def get_codebook(quantizer):
    
    """Given a quantizer returns the learned codebook"""
    
    codebook=quantizer.get_weights()[0]
    return codebook

def codebook_from_index(codebook, k_index):
    
    """"Transform indices into the corresponding codebook entries"""
    K=codebook.shape[0]
    D=codebook.shape[1]
    lookup_ = tf.reshape(codebook, shape=(1, 1, 1,K, D))
    k_index_one_hot = tf.one_hot(k_index,K)
    z_q = lookup_ * k_index_one_hot[..., None]
    z_q = tf.reduce_sum(z_q, axis=-2)
    return z_q
  
def build_indicization_model():
  
    """Load models and wrap them into a single keras model that, givena a batch of images, outputs a tuple
    of their top, mid and bot indices """
    
    top_encoder=loader.load_top_encoder()
    mid_encoder=loader.load_mid_encoder()
    bot_encoder=loader.load_bot_encoder()
    top_quantizer=loader.load_top_quantizer()
    mid_quantizer=loader.load_mid_quantizer()
    bot_quantizer=loader.load_bot_quantizer()
    top_codebook=get_codebook(top_quantizer)
    mid_codebook=get_codebook(mid_quantizer)
    bot_codebook=get_codebook(bot_quantizer)
    
    inp=tf.keras.Input(shape=[img_size,img_size, 3])
    
    top_encoded=top_encoder(inp)
    top_quantized=top_quantizer(top_encoded)
    top_lookup_ = tf.reshape(top_codebook, shape=(1,1,1,KT,DT))
    z_t = tf.expand_dims(top_encoded, -2)
    top_dist = tf.norm(z_t - top_lookup_, axis=-1)
    top_index = tf.argmin(top_dist, axis=-1)
    
    mid_encoded=mid_encoder([top_quantized,inp])
    mid_quantized=mid_quantizer(mid_encoded)
    mid_lookup_ = tf.reshape(mid_codebook, shape=(1,1,1,KM,DM))
    z_m = tf.expand_dims(mid_encoded, -2)
    mid_dist = tf.norm(z_m - mid_lookup_, axis=-1)
    mid_index = tf.argmin(mid_dist, axis=-1)
    
    bot_encoded=bot_encoder([top_quantized, mid_quantized,inp])
    bot_lookup_ = tf.reshape(bot_codebook, shape=(1,1,1,KB,DB))
    z_b = tf.expand_dims(bot_encoded, -2)
    bot_dist = tf.norm(z_b - bot_lookup_, axis=-1)
    bot_index = tf.argmin(bot_dist, axis=-1)
    
    model=tf.keras.Model(inputs=inp, outputs=[top_index,mid_index, bot_index])

    return model

def encode_images_indices(images):
    
    """Given a batch of images, returns their indices"""
    
    indicizer=build_indicization_model()
    top_indices, mid_indices, bot_indices=indicizer.predict(images)
    
    return top_indices, mid_indices, bot_indices

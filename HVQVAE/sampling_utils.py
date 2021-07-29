"""This module contains helpful functions to generate new images"""

from HVQVAE.load_utils import load_decoder, load_top_quantizer, load_mid_quantizer, load_bot_quantizer
from HVQVAE.utils import get_codebook, codebook_from_index
import numpy as np
import tensorflow_probability as tfp
import progressbar

progress_bar=progressbar.progressbar
top_latent_shape=[16,16]
mid_latent_shape=[32,32]
bot_latent_shape=[64,64]

def sampler(codes, prior):
  
    """Sample codes from the prior"""

    fc2=prior.predict(codes)
    dist=tfp.distributions.Categorical(logits=fc2)
    sampled = dist.sample()
    
    return sampled

def sample_from_top_prior(n, top_prior, top_latent_shape=top_latent_shape):

    """Sample from the top prior one pixel at a time from a blank image"""

    codes = np.zeros((n,top_latent_shape[0],top_latent_shape[1]),dtype='int32')
    print('Sampling codes from the top prior...')
    for i in progress_bar(range(codes.shape[1])):
        for j in range(codes.shape[2]):
            sampled = sampler(codes,top_prior)
            codes[:, i, j] = sampled[:, i, j]
    
    return codes



def sample_from_mid_prior(n, mid_prior, top_codes, mid_latent_shape=mid_latent_shape):

    """Sample a batch of n codes from the mid prior one pixel at a time from a 
       blank image and a batch of n top codes"""

    codes = np.zeros((n,mid_latent_shape[0],mid_latent_shape[1]),dtype='int32')
    top_cond=top_codes
    print('Sampling codes from the middle prior...')
    for i in progress_bar(range(codes.shape[1])):
        for j in range(codes.shape[2]):
            sampled = sampler([top_cond,codes],mid_prior)
            codes[:, i, j] = sampled[:, i, j]

    return codes


def sample_from_bottom_prior(n, bot_prior, top_codes, mid_codes, bot_latent_shape=bot_latent_shape):

    """Sample a batch of n codes from the mid prior one pixel at a time from a 
       blank image and a batch of n top and mid codes"""

    codes = np.zeros((n,bot_latent_shape[0],bot_latent_shape[0]),dtype='int32')
    mid_cond=mid_codes
    top_cond=top_codes
    print('Sampling codes from the bottom prior...')
    for i in progress_bar(range(codes.shape[1])):
        for j in range(codes.shape[2]):
            sampled = sampler([top_cond,mid_cond,codes], bot_prior)
            codes[:, i, j] = sampled[:, i, j]
    
    return codes    

def generate(n, top_prior, mid_prior, bot_prior):

    """Generate new images from the codes generated by the priors"""

    top_quantizer=load_top_quantizer()
    mid_quantizer=load_mid_quantizer()
    bot_quantizer=load_bot_quantizer()
    top_codebook=get_codebook(top_quantizer)
    mid_codebook=get_codebook(mid_quantizer)
    bot_codebook=get_codebook(bot_quantizer)

    top_codes=sample_from_top_prior(n, top_prior)
    mid_codes=sample_from_mid_prior(n,mid_prior, top_codes)
    bot_codes=sample_from_bottom_prior(n,bot_prior, top_codes, mid_codes)

    decoder=load_decoder()
    tq=codebook_from_index(top_codebook, top_codes)
    mq=codebook_from_index(mid_codebook, mid_codes)
    bq=codebook_from_index(bot_codebook, bot_codes)
    generated=decoder.predict([tq, mq, bq])

    return generated, top_codes, mid_codes, bot_codes

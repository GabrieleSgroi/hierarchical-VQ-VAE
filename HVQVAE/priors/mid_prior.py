import tensorflow as tf
from HVQVAE.hyperparameters import KT,DT, KM, DM
from HVQVAE.custom_layers import VectorQuantizer, CBAM, GateActivation,  CausalAttentionModule, MaskedConv2D
from HVQVAE.utils import get_codebook, codebook_from_index
from HVQVAE.load_utils import load_top_quantizer, load_mid_quantizer

from tensorflow.keras.layers import Conv2D, Conv2DTranspose, Concatenate, BatchNormalization, Lambda,ZeroPadding2D,Cropping2D, Dropout
from tensorflow.keras import Model, Input
import os

num_blocks =20       # Number of Gated PixelCNN blocks in the architecture

r=1 #Top to bottom halvings (i.e. d_bot=2^r d_top)
mid_latent_shape=[32,32]
top_latent_shape=[16,16]
ACT=tf.keras.layers.ELU(alpha=0.1)

top_quantizer=load_top_quantizer()
top_codebook=get_codebook(top_quantizer) 

mid_quantizer=load_mid_quantizer()
mid_codebook=get_codebook(mid_quantizer)

def shift_pass(v_stack):
    #shift to apply to the vertical pass so that the horizontal stack remains causal
    shift=ZeroPadding2D(padding=((1,0),(0,0)))(v_stack)
    shift=Cropping2D(cropping=((0,1),(0,0)))(shift)
    return shift
    
def gated_block(v_stack_in, h_stack_in, out_dim, conditional, kernel,dilation=1, mask='b', residual=True, i=0):
    """Implementation of the basic gated pixelcnn block following sect. 2 of TEXT- AND STRUCTURE-CONDITIONAL PIXELCNN,
    S. Reed, A. van den Oord, N. Kalchbrenner, V. Bapst, M. Botvinick, N. de Freitas Google DeepMind"""
   
    if residual:
        v_attention=CausalAttentionModule(out_dim,dilation=2, mask_type=mask)(v_stack_in)
        v_stack_in=v_stack_in+v_attention
        h_attention=CausalAttentionModule(out_dim, dilation=2, mask_type=mask)(h_stack_in)
        h_stack_in=h_stack_in+h_attention


    Extraconv=Conv2D(kernel_size=1, filters=out_dim*2,activation=ACT)(v_stack_in)
    v_stack=Conv2D(kernel_size=(1,kernel), filters=out_dim,activation=ACT,dilation_rate=dilation, padding='same')(v_stack_in)
    v_stack = MaskedConv2D( mask,kernel_size=(kernel,1), filters=out_dim * 2,dilation_rate=dilation, padding='same',
                           activation=ACT, name="v_masked_conv_{}".format(i))(v_stack)
    vBN=BatchNormalization(renorm=True)(v_stack+Extraconv+conditional)
    v_stack_out = GateActivation()(vBN)
    
    h_stack = MaskedConv2D( mask,filters=out_dim * 2,dilation_rate=dilation, padding='same',
                           kernel_size=(1,kernel),activation=ACT, name="h_masked_conv_{}".format(i))(h_stack_in)

    shift=shift_pass(v_stack)
    h_stack_1 =Conv2D(filters=out_dim * 2, kernel_size=1, strides=(1, 1),activation=ACT, name="v_to_h_{}".format(i))(shift)
    hBN=BatchNormalization(renorm=True)(h_stack + h_stack_1+conditional)
    h_stack_out =GateActivation()(hBN)
    
    skip=Conv2D(filters=out_dim, kernel_size=1, strides=(1, 1), activation=ACT)(h_stack_out)

    h_stack_out = Conv2D(filters=out_dim, kernel_size=1, strides=(1, 1), activation=ACT, name="res_conv_{}".format(i))(h_stack_out)
    
    if residual:
        h_stack_out += h_stack_in
    
    return v_stack_out, h_stack_out,skip


def build_mid_prior(num_layers=20, num_feature_maps=128):
   pixelcnn_prior_inputs = Input(shape=(mid_latent_shape[0], mid_latent_shape[1]), name='pixelcnn_prior_inputs', dtype=tf.int64)
   Top_input=Input(shape=(top_latent_shape[0], top_latent_shape[1]), name='conditional_input', dtype=tf.int64)
   cq=top_codebook_from_index(top_codebook, Top_input)
   cq=Conv2D(kernel_size=3, filters=num_feature_maps, activation=ACT, padding='same')(cq)
   attention=CBAM(num_feature_maps, activation=ACT, dilation=2, kernel_size=5)(cq)
   cq=Conv2D(kernel_size=3, filters=num_feature_maps, activation=ACT, padding='same')(cq+attention)
   cq=Conv2DTranspose(kernel_size=4, filters=num_feature_maps*2, strides=2, padding='same', activation=ACT)(cq)
   cq=BatchNormalization(renorm=True)(cq)
   z_q =mid_codebook_from_index(mid_codebook, pixelcnn_prior_inputs)
   v_stack_in, h_stack_in = z_q, z_q
   for i in range(0,num_layers):
        if i%5==2:
            dilation=2
        elif i%5==3:
            dilation=3
        else:
            dilation=1
            
        mask = 'b' if i > 0 else 'a'
        kernel_size = 3 if i > 0 else 9
        residual = True if i > 0 else False
        v_stack_in, h_stack_in ,skipped = gated_block(v_stack_in, h_stack_in, num_feature_maps,cq,
                            kernel=kernel_size, mask=mask,dilation=dilation, residual=residual, i=i + 1)
                            
        if i==0:
            skip=skipped
        else:
            skip+=skipped
   
   uplifting=Conv2D(filters= num_feature_maps, kernel_size=1, activation=ACT, padding="same")(cq)
   skip=BatchNormalization(renorm=True)(skip)
   fc = Conv2D(filters=num_feature_maps, kernel_size=1,padding='same', activation=ACT)(skip)
   attention=CausalAttentionModule(num_feature_maps,dilation=2, mask_type='b')(fc)
   fc = Conv2D(filters=num_feature_maps, kernel_size=1,padding='same', activation=ACT)(fc+attention)
   attention=CausalAttentionModule(num_feature_maps,dilation=2,mask_type='b')(fc)
   fc = Conv2D(filters=2*num_feature_maps, kernel_size=1,padding='same', activation=ACT)(fc+attention+uplifting)
   fc=BatchNormalization(renorm=True)(fc)
   fc2 = Conv2D(filters=KM, kernel_size=1, name="fc2")(fc) 

   pixelcnn_prior = Model(inputs=[Top_input,pixelcnn_prior_inputs], outputs=fc2, name='pixelcnn-prior')
 
   return pixelcnn_prior
 



import tensorflow as tf
from HVQVAE.hyperparameters import KT,DT
from HVQVAE.custom_layers import VectorQuantizer, CBAM, GateActivation,  CausalAttentionModule, MaskedConv2D
from HVQVAE.utils import get_codebook, codebook_from_index
from HVQVAE.load_utils import load_top_quantizer 
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, Concatenate, BatchNormalization, Lambda,ZeroPadding2D,Cropping2D, Dropout
from tensorflow.keras import Model, Input
import os

ACT=tf.keras.layers.ELU(alpha=0.1) #Activation function
PIXELCNN_NUM_BLOCKS = 20  # Number of Gated PixelCNN blocks in the architecture
PIXELCNN_NUM_FEATURE_MAPS = 256
latent_shape=[16,16]

top_quantizer=load_top_quantizer()
top_codebook=get_codebook(top_quantizer)


def shift_pass(v_stack):
    #shift to apply to the vertical pass so that the horizontal stack remains causal
    shift=ZeroPadding2D(padding=((1,0),(0,0)))(v_stack)
    shift=Cropping2D(cropping=((0,1),(0,0)))(shift)
    return shift
    
def gated_block(v_stack_in, h_stack_in, out_dim, kernel, mask='b', residual=True, i=0,dilation=1):
    """Implementation of the basic gated pixelcnn block following sect. 2 of TEXT- AND STRUCTURE-CONDITIONAL PIXELCNN,
    S. Reed, A. van den Oord, N. Kalchbrenner, V. Bapst, M. Botvinick, N. de Freitas Google DeepMind"""

    if residual:
        v_attention=CausalAttentionModule(out_dim, mask,dilation=2)(v_stack_in)
        v_stack_in=v_stack_in+v_attention
        h_attention=CausalAttentionModule(out_dim, mask,dilation=2)(h_stack_in)
        h_stack_in=h_stack_in+h_attention


    Extraconv=Conv2D(kernel_size=1, filters=out_dim*2,activation=ACT)(v_stack_in)
    v_stack=Conv2D(kernel_size=(1,kernel), filters=out_dim,activation=ACT, dilation_rate=dilation, padding="same")(v_stack_in)
    v_stack = MaskedConv2D(mask, filters=out_dim*2, kernel_size=(kernel, 1),padding="same", dilation_rate=dilation, activation=ACT, name="v_masked_conv_{}".format(i))(v_stack)
    vBN=BatchNormalization()(v_stack+Extraconv)
    v_stack_out = GateActivation()(vBN)


    h_stack = MaskedConv2D(mask,filters=out_dim * 2, kernel_size=(1,kernel),activation=ACT,padding="same",dilation_rate=dilation, name="h_masked_conv_{}".format(i))(h_stack_in)
    shift=shift_pass(v_stack)
    h_stack_1 =Conv2D(filters=out_dim * 2, kernel_size=1, activation=ACT, name="v_to_h_{}".format(i))(shift)
    hBN=BatchNormalization()(h_stack + h_stack_1)
    h_stack_out=GateActivation()(hBN)
    
    skip=Conv2D(filters=out_dim, kernel_size=1, strides=(1, 1), activation=ACT)(h_stack_out)

    h_stack_out = Conv2D(filters=out_dim, kernel_size=1, strides=(1, 1), activation=ACT, name="res_conv_{}".format(i))(h_stack_out)
    h_stack_out=Dropout(0.2)(h_stack_out)
    
    if residual:
        h_stack_out += h_stack_in
    
    return v_stack_out, h_stack_out,skip


def build_top_prior(num_layers=PIXELCNN_NUM_BLOCKS, num_feature_maps=PIXELCNN_NUM_FEATURE_MAPS):
    pixelcnn_prior_inputs = Input(shape=(latent_shape[0],latent_shape[1]), name='pixelcnn_prior_inputs', dtype=tf.int64)
    z_q =codebook_from_index(top_codebook, pixelcnn_prior_inputs) # maps indices to the actual codebook
    v_stack_in, h_stack_in = z_q, z_q
    for i in range(0,num_layers):
        mask = 'b' if i > 0 else 'a'
        kernel_size = 3 if i > 0 else 7
        residual = True if i > 0 else False
        
        if i % 5==1 or i %5==3:
            dilation=2
        else:
            dilation=1
            
        v_stack_in, h_stack_in, skipped = gated_block(v_stack_in, h_stack_in, num_feature_maps,
                            kernel=kernel_size, mask=mask, residual=residual,dilation=dilation, i=i + 1)
        if i==0:
            skip=skipped
        else:
            skip+=skipped
            
    skip=BatchNormalization()(skip)
    fc = Conv2D(filters=num_feature_maps, kernel_size=1,padding='same', activation=ACT)(skip)
    fc=Dropout(0.2)(fc)
    attention=CausalAttentionModule(num_feature_maps,mask_type='b')(fc)
    fc = Conv2D(filters=num_feature_maps, kernel_size=1,padding='same', activation=ACT)(fc+attention)
    fc=Dropout(0.2)(fc)
    attention=CausalAttentionModule(num_feature_maps,mask_type='b')(fc)
    fc = Conv2D(filters=2*num_feature_maps, kernel_size=1,padding='same', activation=ACT)(fc+skip+attention)
    fc=BatchNormalization()(fc)
    fc=Dropout(0.2)(fc)
    fc = Conv2D(filters=KT, kernel_size=1, name="fc2")(fc) 
    # outputs logits for probabilities of codebook indices for each cell
    
    pixelcnn_prior = Model(inputs=pixelcnn_prior_inputs, outputs=fc, name='pixelcnn-prior')
    # Distribution to sample from the pixelcnn
    
    return pixelcnn_prior



def load_top_prior():
    top_prior=build_top_prior()
    url='https://github.com/GabrieleSgroi/hierarchical-VQ-VAE/blob/main/HVQVAE/priors/priors_weights/top_prior_weights.h5?raw=true'
    weights_dir=tf.keras.utils.get_file('top_prior_weights.h5',url)
    top_prior.load_weights(weights_dir) 
    print('Top prior loaded')
    
    return top_prior

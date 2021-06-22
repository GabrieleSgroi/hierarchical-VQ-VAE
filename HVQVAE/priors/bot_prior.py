import tensorflow as tf
from HVQVAE.hyperparameters import KT,DT, KM, DM, KB,DB
from HVQVAE.custom_layers import VectorQuantizer, CBAM, GateActivation,  CausalAttentionModule, MaskedConv2D
from HVQVAE.utils import get_codebook, codebook_from_index
from HVQVAE.load_utils import load_bot_quantizer, load_mid_quantizer, load_top_quantizer
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, Concatenate, BatchNormalization, Lambda,ZeroPadding2D,Cropping2D, Dropout
from tensorflow.keras import Model, Input

bot_latent_shape=[32,32]
mid_latent_shape=[32,32]
top_latent_shape=[16,16]
ACT=tf.keras.layers.ELU(alpha=0.1)

top_quantizer=load_top_quantizer()
top_codebook=get_codebook(top_quantizer) 

mid_quantizer=load_mid_quantizer()
mid_codebook=get_codebook(mid_quantizer)


bot_quantizer=load_bot_quantizer()
bot_codebook=get_codebook(bot_quantizer)

def gated_block(v_stack_in, h_stack_in, conditional, out_dim, kernel, mask='b',dilation=1, residual=True, i=0):
    """Basic Gated-PixelCNN block"""
   
    
    Extraconv=Conv2D(kernel_size=1, filters=out_dim*2,activation=ACT)(v_stack_in)
    v_stack=Conv2D(kernel_size=(1,kernel), filters=out_dim,activation=ACT,dilation_rate=dilation,padding='same')(v_stack_in)
    v_stack = MaskedConv2D( mask,kernel_size=(kernel, 1), filters=out_dim*2, dilation_rate=dilation,
                           activation=ACT, padding='same',name="v_masked_conv_{}".format(i))(v_stack)
    
    vBN=BatchNormalization(renorm=True)(v_stack+Extraconv+conditional)
    v_stack_out = GateActivation()(vBN)
    
    h_stack = MaskedConv2D(mask,kernel_size=(1, kernel ),filters=out_dim * 2, padding='same', dilation_rate=dilation,
                           activation=ACT, name="h_masked_conv_{}".format(i))(h_stack_in)
    shift=shift_pass(v_stack)
    h_stack_1 =Conv2D(filters=out_dim * 2, kernel_size=1, strides=(1, 1),activation=ACT, name="v_to_h_{}".format(i))(shift)
    hBN=BatchNormalization(renorm=True)(h_stack + h_stack_1+conditional)
    h_stack_out =GateActivation()(hBN)
    skip=Conv2D(filters=out_dim, kernel_size=1, strides=(1, 1), activation=ACT)(h_stack_out)
    
    h_stack_out = Conv2D(filters=out_dim, kernel_size=1, strides=(1, 1), activation=ACT, name="res_conv_{}".format(i))(h_stack_out)

    if residual:
        h_stack_out += h_stack_in
    
    return v_stack_out, h_stack_out,skip


def build_pixelcnn(num_layers=20, num_feature_maps=64):
    pixelcnn_prior_inputs = Input(shape=(bot_latent_shape[0], bot_latent_shape[1]), name='pixelcnn_prior_inputs', dtype=tf.int64)
    top_input=Input(shape=(top_latent_shape[0], top_latent_shape[1]), name='conditional_top_input', dtype=tf.int64) #top-level input indices
    ct_q=codebook_from_index(top_codebook, top_input) # maps indices to the actual codebook entries
    mid_input=Input(shape=(mid_latent_shape[0],mid_latent_shape[1]), name='conditional_mid_input', dtype=tf.int64) #mid-evel input indices
    cm_q=codebook_from_index(mid_codebook, mid_input) # maps indices to the actual codebook entries
    z_q =codebook_from_index(bot_codebook, pixelcnn_prior_inputs) # maps indices to the actual codebook entries
    ct_q=Conv2DTranspose(kernel_size=2, filters=num_feature_maps, strides=2, padding='same', activation=ACT)(ct_q)
    conditional=Concatenate(axis=-1)([ct_q, cm_q])
    conditional=Conv2DTranspose(kernel_size=2, filters=2*num_feature_maps, strides=2, padding='same', activation=ACT)(conditional)

    v_stack_in, h_stack_in = z_q, z_q
    for i in range(0,num_layers):
        mask = 'b' if i > 0 else 'a'
        kernel_size = 5 if i > 0 else 9
        residual = True if i > 0 else False
        if i%5==2:
            dilation=2
        elif i%5==3:
            dilation=3
        else:
            dilation=1
        v_stack_in, h_stack_in, skipped = gated_block(v_stack_in, h_stack_in, conditional, num_feature_maps,dilation=dilation,
                            kernel=kernel_size,mask=mask, residual=residual, i=i + 1) #fix masking also in other priors
        if i==0:
            skip=skipped
        else:
            skip+=skipped
    
    uplifting=Conv2D(filters= num_feature_maps, kernel_size=1, activation=ACT, padding="same")(conditional)
    skip=BatchNormalization(renorm=True)(skip)
    fc = Conv2D(filters=num_feature_maps, kernel_size=1,padding='same', activation=ACT)(skip)
    attention=CausalAttentionModule(num_feature_maps,mask_type='b', dilation=4)(fc)
    fc = Conv2D(filters=num_feature_maps, kernel_size=1,padding='same', activation=ACT)(fc+attention)
    attention=CausalAttentionModule(num_feature_maps,mask_type='b', dilation=3)(fc)
    fc = Conv2D(filters=num_feature_maps, kernel_size=1,padding='same', activation=ACT)(fc+attention)
    attention=CausalAttentionModule(num_feature_maps,mask_type='b', dilation=2)(fc)
    fc = Conv2D(filters=2*num_feature_maps, kernel_size=1,padding='same', activation=ACT)(fc+attention+uplifting)
    fc=BatchNormalization(renorm=True)(fc)
    fc = Conv2D(filters=KB, kernel_size=1)(fc)  
    # outputs logits for probabilities of codebook indices for each cell
    
    pixelcnn_prior = Model(inputs=[top_input, mid_input, pixelcnn_prior_inputs], outputs=fc, name='pixelcnn-prior')
 
    return pixelcnn_prior

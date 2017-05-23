#!/usr/bin/env python
from .layers import Model
import tensorflow as tf

def down_scale_images(input_images, K):
    _, h, w, _ = input_images.shape.as_list()
    downsampled = tf.image.resize_area(input_images, [h//K, w//K])
    return downsampled

def generator_subpixel_net(gene_input, ratio, batch_size = None, channels = 3):
    """Upscale input images with ratio times
    """
    mapsize = 3
    res_units  = [256, 128, 96]

    old_vars = tf.global_variables()

    # See Arxiv 1603.05027
    model = Model.Model('GEN_subpixel', gene_input, batch_size)
    with tf.variable_scope('GEN_subpixel'):
        for ru in range(len(res_units)-1):
            nunits  = res_units[ru]
    
            for j in range(2):
                model.add_residual_block(nunits, mapsize=mapsize)
    
            # Spatial upscale (see http://distill.pub/2016/deconv-checkerboard/)
            # and transposed convolution
            # TODO: nn
            # model.add_upscale()
            
            model.add_batch_norm()
            model.add_relu()
            model.add_conv2d_transpose(nunits, mapsize=mapsize, stride=1, stddev_factor=1.)
    
        # Finalization a la "all convolutional net"
        nunits = res_units[-1]
        model.add_conv2d(nunits, mapsize=mapsize, stride=1, stddev_factor=2.)
        # Worse: model.add_batch_norm()
        model.add_relu()
    
        model.add_conv2d(nunits, mapsize=1, stride=1, stddev_factor=2.)
        # Worse: model.add_batch_norm()
        model.add_relu()
    
        # Last layer is sigmoid with no batch normalization
        # upscaling ratio
        # TODO: nn
        # model.add_conv2d(channels, mapsize=1, stride=1, stddev_factor=1.)


        # TODO: subpixel
        # Spatial upscale : subpixel
        model.add_conv2d(channels * ratio * ratio, mapsize=1, stride=1, stddev_factor=1.)
        model.add_upscale_subpixel(r = ratio, color=True)

        # model.add_tanh()
        model.add_sigmoid()
        
        new_vars  = tf.global_variables()
        gene_vars_list = list(set(new_vars) - set(old_vars))

    return model.get_output(), gene_vars_list
    


def main():
    pass

if __name__ == '__main__':
    main()

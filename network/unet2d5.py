#   CODE ADAPTED FROM: https://github.com/MIC-DKFZ/nnUNet/blob/master/nnunet/network_architecture/generic_UNet.py

from torch import nn
import torch
import numpy as np
import torch.nn.functional


class InitWeights_He(object):
    def __init__(self, neg_slope=1e-2):
        self.neg_slope = neg_slope

    def __call__(self, module):
        if isinstance(module, nn.Conv3d) or isinstance(module, nn.Conv2d) or isinstance(module, nn.ConvTranspose2d) or isinstance(module, nn.ConvTranspose3d):
            module.weight = nn.init.kaiming_normal_(module.weight, a=self.neg_slope)
            if module.bias is not None:
                module.bias = nn.init.constant_(module.bias, 0)


class ConvNormNonlinBlock(nn.Module):
    def __init__(
        self, 
        input_channels, 
        output_channels,
        conv_op=nn.Conv3d, 
        conv_kwargs=None,
        norm_op=nn.InstanceNorm3d, 
        norm_op_kwargs=None,
        nonlin=nn.LeakyReLU, 
        nonlin_kwargs=None):

        """
        Block: Conv->Norm->Activation->Conv->Norm->Activation
        """

        super(ConvNormNonlinBlock, self).__init__()

        self.nonlin_kwargs = nonlin_kwargs
        self.nonlin = nonlin
        self.conv_op = conv_op
        self.norm_op = norm_op
        self.norm_op_kwargs = norm_op_kwargs
        self.conv_kwargs = conv_kwargs
        self.output_channels = output_channels

        self.first_conv = self.conv_op(input_channels, output_channels, **self.conv_kwargs)
        self.first_norm = self.norm_op(output_channels, **self.norm_op_kwargs)
        self.first_acti = self.nonlin(**self.nonlin_kwargs)

        self.second_conv = self.conv_op(output_channels, output_channels, **self.conv_kwargs)
        self.second_norm = self.norm_op(output_channels, **self.norm_op_kwargs)
        self.second_acti = self.nonlin(**self.nonlin_kwargs)        

        self.block = nn.Sequential(
            self.first_conv,
            self.first_norm,
            self.first_acti,
            self.second_conv,
            self.second_norm,
            self.second_acti
            )


    def forward(self, x):
        return self.block(x)



class Upsample(nn.Module):
    def __init__(self, size=None, scale_factor=None, mode='nearest', align_corners=False):
        super(Upsample, self).__init__()
        self.align_corners = align_corners
        self.mode = mode
        self.scale_factor = scale_factor
        self.size = size

    def forward(self, x):
        return nn.functional.interpolate(x, size=self.size, scale_factor=self.scale_factor, mode=self.mode, align_corners=self.align_corners)


class U_Net2D5(nn.Module):

    def __init__(
        self, 
        input_channels=1, 
        base_num_features=16, 
        num_classes=2, 
        num_pool=4,
        conv_op=nn.Conv3d,
        conv_kernel_sizes=None,
        norm_op=nn.InstanceNorm3d, 
        norm_op_kwargs=None,
        nonlin=nn.LeakyReLU, 
        nonlin_kwargs=None,
        weightInitializer=InitWeights_He(1e-2)):
        """
        2.5D CNN combining 2D and 3D convolutions to dealwith the low through-plane resolution.
        The first two stages have 2D convolutions while the others have 3D convolutions. 

        Architecture inspired by: 
        Wang,et al: Automatic segmentation of  vestibular  schwannoma  from  t2-weighted  mri  
        by  deep  spatial  attention  with hardness-weighted loss. MICCAI 2019. 
        """
        super(U_Net2D5, self).__init__()


        if nonlin_kwargs is None:
             nonlin_kwargs = {'negative_slope':1e-2, 'inplace':True}
    
        if norm_op_kwargs is None:
            norm_op_kwargs = {'eps':1e-5, 'affine':True, 'momentum':0.1}

        self.conv_kwargs = {'stride':1, 'dilation':1, 'bias':True}

        self.nonlin = nonlin
        self.nonlin_kwargs = nonlin_kwargs
        self.norm_op_kwargs = norm_op_kwargs
        self.weightInitializer = weightInitializer
        self.conv_op = conv_op
        self.norm_op = norm_op
        self.num_classes = num_classes

        upsample_mode = 'trilinear'
        pool_op = nn.MaxPool3d
        pool_op_kernel_sizes = [(2, 2, 2)] * num_pool
        if conv_kernel_sizes is None:
            conv_kernel_sizes = [(3, 3, 1)] * 2 + [(3,3,3)]*(num_pool - 1)


        self.input_shape_must_be_divisible_by = np.prod(pool_op_kernel_sizes, 0, dtype=np.int64)
        self.pool_op_kernel_sizes = pool_op_kernel_sizes
        self.conv_kernel_sizes = conv_kernel_sizes

        self.conv_pad_sizes = []
        for krnl in self.conv_kernel_sizes:
            self.conv_pad_sizes.append([1 if i == 3 else 0 for i in krnl])

        self.conv_blocks_context = []
        self.conv_blocks_localization = []
        self.td = []
        self.tu = []
        self.seg_outputs = []


        input_features = input_channels 
        output_features = base_num_features


        for d in range(num_pool):
            self.conv_kwargs['kernel_size'] = self.conv_kernel_sizes[d]
            self.conv_kwargs['padding'] = self.conv_pad_sizes[d]
            # add convolutions
            self.conv_blocks_context.append(ConvNormNonlinBlock(input_features, output_features,
                                                                self.conv_op, self.conv_kwargs, self.norm_op,
                                                                self.norm_op_kwargs, self.nonlin, self.nonlin_kwargs))

            self.td.append(pool_op(pool_op_kernel_sizes[d]))
            input_features = output_features
            output_features = 2* output_features # Number of kernel increases by a factor 2 after each pooling


        final_num_features = self.conv_blocks_context[-1].output_channels
        self.conv_kwargs['kernel_size'] = self.conv_kernel_sizes[num_pool]
        self.conv_kwargs['padding'] = self.conv_pad_sizes[num_pool]
        self.conv_blocks_context.append(ConvNormNonlinBlock(input_features, final_num_features,
                                                            self.conv_op, self.conv_kwargs, self.norm_op,
                                                            self.norm_op_kwargs, self.nonlin, self.nonlin_kwargs))


        # now lets build the localization pathway
        for u in range(num_pool):
            nfeatures_from_skip = self.conv_blocks_context[-(2 + u)].output_channels # self.conv_blocks_context[-1] is bottleneck, so start with -2
            n_features_after_tu_and_concat = nfeatures_from_skip * 2

            # the first conv reduces the number of features to match those of skip
            # the following convs work on that number of features
            # if not convolutional upsampling then the final conv reduces the num of features again
            if u != num_pool - 1:
                final_num_features = self.conv_blocks_context[-(3 + u)].output_channels
            else:
                final_num_features = nfeatures_from_skip

            self.tu.append(Upsample(scale_factor=pool_op_kernel_sizes[-(u+1)], mode=upsample_mode))


            self.conv_kwargs['kernel_size'] = self.conv_kernel_sizes[-(u+1)]
            self.conv_kwargs['padding'] = self.conv_pad_sizes[-(u+1)]
            self.conv_blocks_localization.append(ConvNormNonlinBlock(n_features_after_tu_and_concat, final_num_features,
                                                            self.conv_op, self.conv_kwargs, self.norm_op,
                                                            self.norm_op_kwargs, self.nonlin, self.nonlin_kwargs))


        self.final_conv = conv_op(self.conv_blocks_localization[-1].output_channels, num_classes, 1, 1, 0, 1, 1, False)



        # register all modules properly
        self.conv_blocks_localization = nn.ModuleList(self.conv_blocks_localization)
        self.conv_blocks_context = nn.ModuleList(self.conv_blocks_context)
        self.td = nn.ModuleList(self.td)
        self.tu = nn.ModuleList(self.tu)

        if self.weightInitializer is not None:
            self.apply(self.weightInitializer)

    def forward(self, x):
        skips = []
        seg_outputs = []
        for d in range(len(self.conv_blocks_context) - 1):
            x = self.conv_blocks_context[d](x)
            skips.append(x)
            x = self.td[d](x)

        x = self.conv_blocks_context[-1](x)

        for u in range(len(self.tu)):
            x = self.tu[u](x)
            x = torch.cat((x, skips[-(u + 1)]), dim=1)
            x = self.conv_blocks_localization[u](x)
        
        output = self.final_conv(x)
        return output
                





import torch
import torch.nn as nn
from fvcore.common.registry import Registry
from mmcv.cnn import ConvModule
from utils.decoder_utils import MLP, resize
import hydra


DECODER = Registry('Decoder')


@DECODER.register()
class QueryType(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, img_seqs, txt_seqs=None, txt_pad_masks=None, train=True):
        pass


@DECODER.register()
class LabelType(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, joint_seqs):
        pass


# @DECODER.register()
# class DenseType(nn.Module):
#     def __init__(self):
#         super().__init__()
        
    
#     def forward(self, joint_seqs):
#         pass

#==================================================================


@DECODER.register()
class DenseType(nn.Module):
    """
    SegFormer Decoder
    """
    def __init__(self, info):
        super().__init__()

        self.info = info
        c1_in_channels, c2_in_channels, c3_in_channels, c4_in_channels = self.info.in_channels
        embed_dim = self.info.embed_dim

        self.linear_c4 = MLP(input_dim=c4_in_channels, embed_dim=embed_dim)
        self.linear_c3 = MLP(input_dim=c3_in_channels, embed_dim=embed_dim)
        self.linear_c2 = MLP(input_dim=c2_in_channels, embed_dim=embed_dim)
        self.linear_c1 = MLP(input_dim=c1_in_channels, embed_dim=embed_dim)

        self.linear_fuse = ConvModule(
            in_channels=embed_dim*4,
            out_channels=embed_dim,
            kernel_size=1,
            norm_cfg=dict(type='SyncBN', requires_grad=True)
        )

        self.dropout = nn.Dropout2d(self.info.dropout_ratio)
        self.linear_pred = nn.Conv2d(embed_dim, self.info.num_classes, kernel_size=1)


    def forward(self, inputs):
        x = self._transform_inputs(inputs, self.info)  # len=4, 1/4,1/8,1/16,1/32
        c1, c2, c3, c4 = x

        ############## MLP decoder on C1-C4 ###########
        n, _, _, _ = c4.shape

        _c4 = self.linear_c4(c4).permute(0,2,1).reshape(n, -1, c4.shape[2], c4.shape[3])
        _c4 = resize(_c4, size=c1.size()[2:],mode='bilinear')

        _c3 = self.linear_c3(c3).permute(0,2,1).reshape(n, -1, c3.shape[2], c3.shape[3])
        _c3 = resize(_c3, size=c1.size()[2:],mode='bilinear')

        _c2 = self.linear_c2(c2).permute(0,2,1).reshape(n, -1, c2.shape[2], c2.shape[3])
        _c2 = resize(_c2, size=c1.size()[2:],mode='bilinear')

        _c1 = self.linear_c1(c1).permute(0,2,1).reshape(n, -1, c1.shape[2], c1.shape[3])

        _c = self.linear_fuse(torch.cat([_c4, _c3, _c2, _c1], dim=1))

        x = self.dropout(_c)
        x = self.linear_pred(x)
        
        x = resize(x, size=self.info.output_resolution, mode='bilinear')

        return x
    

    def _transform_inputs(self, inputs, info):
        """Transform inputs for decoder.
        Args:
            inputs (list[Tensor]): List of multi-level img features.
        Returns:
            Tensor: The transformed inputs
        """
        inputs = [inputs[i] for i in info.in_index]

        return inputs


def build_decoder(decoder_type):
    assert decoder_type in ['QueryType', 'LabelType', 'DenseType']
    return DECODER.get(decoder_type)



@hydra.main(config_path='../configs/decoder', config_name='DenseType')
def main(cfg):
    dense = DenseType(cfg.info)
    in_channels = [256, 512, 1024, 2048]
    r = [224/4, 224/8, 224/16, 224/32]
    
    temp = [torch.randn((8,in_channels[0],r[0],r[0])),
            torch.randn((8,in_channels[1],r[1],r[1])),
            torch.randn((8,in_channels[2],r[2],r[2])),
            torch.randn((8,in_channels[3],r[3],r[3]))]

    x = dense(temp)
    print(x.shape)

if __name__ == '__main__':
    main()
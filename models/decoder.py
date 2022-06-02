import torch
import torch.nn as nn
from torch.nn import functional as F
from .transformer import build_transformer_encoder
from .positional_embedding import build_positional_embedding
from fvcore.common.registry import Registry
from mmcv.cnn import ConvModule
from .decoder_utils import LabelMLP, DenseMLP, resize
from .rec_decoder.modules import Upsample, AttnBlock, Normalize, nonlinearity
from .rec_decoder.modules import ResnetBlock as PVQVAEResnetBlock
from .rec_decoder.rand_transformer import RandTransformer
from .rec_decoder.auto_encoder import PVQVAE

import hydra


DECODER = Registry('Decoder')


@DECODER.register()
class LabelType(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.v_proj = nn.Linear(cfg.input_dim_list[-1], cfg.hidden_dim)
        
        self.block_1 = build_transformer_encoder(cfg.transformer)

        # LabelMLP is a MLP written for the LabelType decoder
        self.output_head = LabelMLP(cfg.transformer.hidden_dim, cfg.num_classes)

        self.pos_embed = build_positional_embedding(
            type=cfg.positional_embedding.type,
            shape=(cfg.image_size//cfg.reduction[-1], cfg.image_size//cfg.reduction[-1]),
            hidden_dim=cfg.positional_embedding.hidden_dim
        )
        self.label_token = nn.Parameter(0.1*torch.randn(cfg.hidden_dim))
    
    def forward(self, v_feature_list, txt_seqs=None, txt_pad_masks=None):
        img_seqs = v_feature_list[-1]   # get last feature
        B, C, h, w = img_seqs.shape
        
        # [B, C, h, w] -> [hw, B, C]
        img_seqs = img_seqs.flatten(2).permute(2, 0, 1)
        img_seqs = self.v_proj(img_seqs)

        label_token = self.label_token.reshape(1, 1, -1).repeat(1, B, 1)
        img_seqs = torch.cat([label_token, img_seqs], dim=0)

        img_masks = torch.zeros((B, 1+h*w), dtype=int).to(self.label_token.device)
        pos_embed = self.pos_embed(B).permute(1, 0, 2).to(self.label_token.device)
        pos_embed = torch.cat([
            torch.zeros_like(self.label_token).view(1, 1, -1).repeat(1, B, 1), pos_embed
        ], dim=0)

        if txt_seqs is not None:
            txt_seqs = txt_seqs.permute(1, 0, 2)   # [B, T, C] -> [T, B, C]
            img_seqs = torch.cat([img_seqs, txt_seqs], dim=0)
            txt_pad_masks = ~txt_pad_masks
            img_masks = torch.cat([img_masks, txt_pad_masks], dim=1)
            pos_embed = torch.cat([pos_embed, torch.zeros_like(txt_seqs)], dim=0)
        
        memory = self.block_1(src=img_seqs, src_key_padding_mask=img_masks, pos=pos_embed)
        label_output = memory[0]   # first token
        label_output = self.output_head(label_output)
        # [B, classes]
        return label_output


@DECODER.register()
class DenseType(nn.Module):
    """
    SegFormer Decoder
    """
    def __init__(self, cfg):
        super().__init__()

        self.output_size = cfg.image_size
        c1_in_channels, c2_in_channels, c3_in_channels, c4_in_channels = cfg.input_dim_list[-4:]
        embed_dim = cfg.hidden_dim

        # DenseMLP is a MLP written for the DenseType decoder
        self.linear_c4 = DenseMLP(input_dim=c4_in_channels, embed_dim=embed_dim)
        self.linear_c3 = DenseMLP(input_dim=c3_in_channels, embed_dim=embed_dim)
        self.linear_c2 = DenseMLP(input_dim=c2_in_channels, embed_dim=embed_dim)
        self.linear_c1 = DenseMLP(input_dim=c1_in_channels, embed_dim=embed_dim)

        self.linear_fuse = ConvModule(
            in_channels=embed_dim*4,
            out_channels=embed_dim,
            kernel_size=1,
            norm_cfg=dict(type='BN', requires_grad=True)
        )

        self.dropout = nn.Dropout2d(cfg.dropout)
        self.linear_pred = nn.Conv2d(embed_dim, cfg.num_classes, kernel_size=1)

    def forward(self, v_feature_list, txt_seqs=None, txt_pad_masks=None):
        c1, c2, c3, c4 = v_feature_list[-4:]

        ############## MLP decoder on C1-C4 ###########
        B = c4.shape[0]

        _c4 = self.linear_c4(c4).permute(0, 2, 1).reshape(B, -1, c4.shape[2], c4.shape[3])
        _c4 = resize(_c4, size=c1.size()[2:], mode='bilinear')

        _c3 = self.linear_c3(c3).permute(0, 2, 1).reshape(B, -1, c3.shape[2], c3.shape[3])
        _c3 = resize(_c3, size=c1.size()[2:], mode='bilinear')

        _c2 = self.linear_c2(c2).permute(0, 2, 1).reshape(B, -1, c2.shape[2], c2.shape[3])
        _c2 = resize(_c2, size=c1.size()[2:], mode='bilinear')

        _c1 = self.linear_c1(c1).permute(0, 2, 1).reshape(B, -1, c1.shape[2], c1.shape[3])

        _c = self.linear_fuse(torch.cat([_c4, _c3, _c2, _c1], dim=1))

        x = self.dropout(_c)
        x = self.linear_pred(x)
        
        x = resize(x, size=self.output_size, mode='bilinear')
        # [B, C, H, W]
        return x

@DECODER.register()
class RecDecoder(nn.Module):
    """
    Sepcific Decoder for 3D reconstruction
    """
    def __init__(self, cfg):
        super().__init__()

        ntoken = 512
        self.dz = self.hz = self.wz = self.grid_size = cfg.tf_cfg.pe.zq_dim

        c1_in_channels, c2_in_channels, c3_in_channels, c4_in_channels = cfg.input_dim_list[-4:]
        self.backbone_output_size = cfg.image_size // cfg.reduction[-1]

        self.linear_to3d = nn.Linear( self.backbone_output_size ** 2, self.dz * self.hz * self.wz)
        self.reduce_channel = torch.nn.Conv3d(c4_in_channels, ntoken, 1)

        nblocks = 3
        use_attn = True
        convt_layers = []
        in_c = 512
        for i in range(nblocks):
            out_c = min(in_c // 2, ntoken)
            convt_layers.append(
                PVQVAEResnetBlock(in_channels=in_c, out_channels=out_c, temb_channels=0, dropout=0.1)
            )
            if use_attn:
                convt_layers.append( AttnBlock(out_c) )
            in_c = out_c

        self.convt_layers = nn.Sequential(*convt_layers)

        self.convt3 = PVQVAEResnetBlock(in_channels=in_c, out_channels=in_c, temb_channels=0, dropout=0.1)
        self.attn3 = AttnBlock(in_c)
        self.norm_out = Normalize(in_c)
        self.conv_out = torch.nn.Conv3d(in_c, ntoken, 1)

        # inference with autosdf prior
        self.tf = RandTransformer(cfg.tf_cfg, vq_conf=cfg.vq_cfg)
        self.tf.eval()
        mparam = cfg.vq_cfg.model.params
        n_embed = mparam.n_embed
        embed_dim = mparam.embed_dim
        ddconfig = mparam.ddconfig
        self.vqvae = PVQVAE(ddconfig, n_embed, embed_dim)
        self.vqvae.eval()
        
        # load state dict
        state_dict = torch.load(cfg.tf_cfg.ckpt)
        self.tf.load_state_dict(state_dict['tf'])
        self.vqvae.load_state_dict(state_dict['vqvae'])

        for p in self.tf.parameters():
            p.requires_grad_(False)
        for p in self.vqvae.parameters():
            p.requires_grad_(False)

        self.sos = 0 # start token
        self.grid_table = self.init_grid(pos_dim=cfg.tf_cfg.pe.pos_dim, zq_dim=self.grid_size).cuda()
    
    def init_grid(self, pos_dim=3, zq_dim=8):
        x = torch.linspace(-1, 1, zq_dim)
        y = torch.linspace(-1, 1, zq_dim)
        if pos_dim == 3:
            z = torch.linspace(-1, 1, zq_dim)
            grid_x, grid_y, grid_z = torch.meshgrid(x, y, z)
            grid = torch.stack([grid_x, grid_y, grid_z], dim=-1)
            pos_sos = torch.tensor([-1., -1., -1-2/zq_dim]).float().unsqueeze(0)
        else:
            grid_x, grid_y = torch.meshgrid(x, y)
            grid = torch.stack([grid_x, grid_y], dim=-1)
            pos_sos = torch.tensor([-1., -1-2/zq_dim]).float().unsqueeze(0)

        grid_table = grid.view(-1, pos_dim)
        grid_table = torch.cat([pos_sos, grid_table], dim=0)
        return grid_table

    def forward(self, v_feature_list, txt_seqs=None, txt_pad_masks=None):
        c1, c2, c3, c4 = v_feature_list[-4:]

        B = c4.shape[0]
        x = self.linear_to3d(c4.reshape(B, -1, c4.shape[2] * c4.shape[3]))
        x = self.reduce_channel(x.reshape(B, -1, self.dz, self.hz, self.wz)) # <B, 512, 8, 8, 8>

        temb = None
        x = self.convt_layers(x)

        x = self.convt3(x, temb)
        if hasattr(self, 'attn3'):
            x = self.attn3(x)
            
        x = self.norm_out(x)
        x = nonlinearity(x)
        x = self.conv_out(x)

        return x
    
    def inference(self, img_logits, topk=30, alpha=0.7, rand_order=False):
        """
        Only used for evaluting.
        """
        def top_k_logits(logits, k=5):
            v, ix = torch.topk(logits, k)
            out = logits.clone()
            out[out < v[:, :, [-1]]] = -float('Inf')
            return out
        
        B = img_logits.shape[0]
        T = self.grid_size ** 3 + 1 # +1 since <sos>

        # gen image prior
        img_logprob = F.log_softmax(img_logits, dim=1) # compute the prob. of next ele
        prob = img_logprob.reshape(self.dz * self.hz * self.wz, B, -1)

        # gen autosdf prior
        seq_len = 1
        if rand_order:
            gen_order = torch.randperm(self.grid_size ** 3).cuda()
        else:
            gen_order = torch.arange(self.grid_size ** 3).cuda()
        prob = prob[gen_order]
        prob = torch.cat([prob[:1], prob])

        pos_shuffled = torch.cat([self.grid_table[:1], self.grid_table[1:][gen_order]], dim=0)
        inp_pos_embedding = pos_shuffled[:-1].clone()
        tgt_pos_embedding = pos_shuffled[1:].clone()

        # auto-regressively gen
        pred = torch.LongTensor(1, B).fill_(self.sos).cuda()
        for t in range(seq_len, T):
            inp = pred
            inp_pos = inp_pos_embedding[:t]
            tgt_pos = tgt_pos_embedding[:t]
            
            outp = self.tf(inp, inp_pos, tgt_pos)
            outp_t = outp[-1:]
            outp_t = F.log_softmax(outp_t, dim=-1)

            outp_t = (1-alpha) * outp_t + alpha * prob[t:t+1] # fuse image prior and autosdf prior

            if topk is not None:
                outp_t = top_k_logits(outp_t, k=topk)

            outp_t = F.softmax(outp_t, dim=-1) # compute prob
            outp_t = outp_t.reshape(B, -1)
            # pred_t = torch.multinomial(outp_t, num_samples=1).squeeze(1)
            pred_t = torch.argmax(outp_t, dim=-1) # directly use the idx with largest probability
            pred_t = pred_t.reshape(1, B)
            pred = torch.cat([pred, pred_t], dim=0)

        pred = pred[1:][torch.argsort(gen_order)] # exclude pred[0] since it's <sos>, <512, B>
        return self.vqvae.decode_enc_idices(pred, z_spatial_dim=self.grid_size) # <B, 1, 64, 64, 64>
    
    def decode_codeidx(self, codeix):
        B = codeix.shape[0]
        codeix = codeix.reshape(B, -1).permute(1, 0)
        return self.vqvae.decode_enc_idices(codeix, z_spatial_dim=self.grid_size) # <B, 1, 64, 64, 64>

def build_decoder(cfg):
    # assert cfg.key in ['QueryType', 'LabelType', 'DenseType']
    assert cfg.key in ['LabelType', 'DenseType', 'RecDecoder']
    return DECODER.get(cfg.key)(cfg.params)


# @DECODER.register()
# class QueryType(nn.Module):
#     def __init__(self, cfg):
#         super().__init__()

#         self.v_proj = []
#         for input_dim in cfg.input_dim_list:
#             self.v_proj.append(nn.Linear(input_dim, cfg.hidden_dim))
        
#         self.block_1 = build_transformer_encoder(cfg.transformer_encoder)
#         self.block_2 = build_transformer_decoder(cfg.transformer_decoder)

#         self.pos_embed = build_positional_embedding(
#             cfg.positional_embedding.type, shape=None, hidden_dim=cfg.positional_embedding.hidden_dim
#         )
#         self.query_embed = nn.Embedding(cfg.num_queries, cfg.hidden_dim)
    
#     def forward(self, v_feature_list, txt_seqs=None, txt_pad_masks=None, train=True):
#         img_seqs = v_feature_list[-1]   # get last feature
#         B, C, h, w = img_seqs.shape
        
#         # [B, C, h, w] -> [hw, B, C]
#         img_seqs = img_seqs.flatten(2).permute(2, 0, 1)
#         img_seqs = self.v_proj[-1](img_seqs)

#         img_masks = torch.zeros((B, h*w), dtype=int)
#         pos_embed = self.pos_embed(B).permute(1, 0, 2)

#         if txt_seqs is not None:
#             txt_seqs = txt_seqs.permute(1, 0, 2)   # [B, T, C] -> [T, B, C]
#             img_seqs = torch.cat([img_seqs, txt_seqs], dim=0)
#             txt_pad_masks = ~txt_pad_masks
#             img_masks = torch.cat([img_masks, txt_pad_masks], dim=1)
#             pos_embed = torch.cat([pos_embed, torch.zeros_like(txt_seqs)], dim=0)
        
#         memory = self.block_1(src=img_seqs, src_key_padding_mask=img_masks, pos=pos_embed)

#         query_embed = self.query_embed.weight.unsqueeze(1).repeat(1, B, 1)
#         hs = self.block_2(
#             tgt=torch.zeros_like(query_embed), memory=memory,
#             memory_key_padding_mask=img_masks, pos=pos_embed, query_pos=query_embed
#         ).permute(1, 0, 2)


@hydra.main(config_path='../configs/decoder', config_name='DenseType')
def main(cfg):
    dense = DenseType(cfg.info)
    in_channels = [256, 512, 1024, 2048]
    r = [int(224/4), int(224/8), int(224/16), int(224/32)]
    
    temp = [torch.randn((8,in_channels[0],r[0],r[0])),
            torch.randn((8,in_channels[1],r[1],r[1])),
            torch.randn((8,in_channels[2],r[2],r[2])),
            torch.randn((8,in_channels[3],r[3],r[3]))]

    temp = [tmp.cuda() for tmp in temp]
    dense.cuda()

    x = dense(temp)
    print(x.shape)

if __name__ == '__main__':
    main()

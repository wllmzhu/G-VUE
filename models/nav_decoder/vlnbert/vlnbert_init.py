# Recurrent VLN-BERT, 2020, by Yicong.Hong@anu.edu.au

from transformers import RobertaConfig, RobertaTokenizer

def get_tokenizer(cfg):
    if cfg.model.decoder.type == 'prevalent':
        tokenizer_class = RobertaTokenizer
        tokenizer = tokenizer_class.from_pretrained(cfg.model.l_backbone.cfg_dir)
    return tokenizer

def get_vlnbert_models(cfg, config=None):
    config_class = RobertaConfig
    if cfg.model.decoder.type == 'prevalent':
        from models.nav_decoder.vlnbert.vlnbert_PREVALENT import VLNBert
        model_class = VLNBert
        model_name_or_path = cfg.train.pretrained_path.prevalent
        vis_config = config_class.from_pretrained(cfg.model.l_backbone.cfg_dir)
        vis_config.img_feature_dim = cfg.model.v_sizes[-1] + 128
        vis_config.img_feature_type = ""
        vis_config.vl_layers = 4
        vis_config.la_layers = 9

        # visual_model = model_class.from_pretrained(model_name_or_path, config=vis_config, cfg=cfg)
        visual_model = model_class(config=vis_config, cfg=cfg)


    return visual_model

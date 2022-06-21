# Recurrent VLN-BERT, 2020, by Yicong.Hong@anu.edu.au

from transformers import (BertConfig, BertTokenizer)

def get_tokenizer(cfg):
    if cfg.model.decoder.type == 'oscar':
        tokenizer_class = BertTokenizer
        model_name_or_path = 'Oscar/pretrained_models/base-no-labels/ep_67_588997'
        tokenizer = tokenizer_class.from_pretrained(model_name_or_path, do_lower_case=True)
    elif cfg.model.decoder.type == 'prevalent':
        tokenizer_class = BertTokenizer
        tokenizer = tokenizer_class.from_pretrained('bert-base-uncased')
    return tokenizer

def get_vlnbert_models(cfg, config=None):
    config_class = BertConfig

    if cfg.model.decoder.type == 'oscar':
        from vlnbert.vlnbert_OSCAR import VLNBert
        model_class = VLNBert
        model_name_or_path = cfg.train.pretrained_path.oscar
        vis_config = config_class.from_pretrained(model_name_or_path, num_labels=2, finetuning_task='vln-r2r')

        vis_config.model_type = 'visual'
        vis_config.finetuning_task = 'vln-r2r'
        vis_config.hidden_dropout_prob = 0.3
        vis_config.hidden_size = 768
        vis_config.img_feature_dim = cfg.model.v_sizes[-1] + 128
        vis_config.num_attention_heads = 12
        vis_config.num_hidden_layers = 12
        visual_model = model_class.from_pretrained(model_name_or_path, from_tf=False, config=vis_config)

    elif cfg.model.decoder.type == 'prevalent':
        from models.nav_decoder.vlnbert.vlnbert_PREVALENT import VLNBert
        model_class = VLNBert
        model_name_or_path = cfg.train.pretrained_path.prevalent
        vis_config = config_class.from_pretrained('bert-base-uncased')
        vis_config.img_feature_dim = cfg.model.v_sizes[-1] + 128
        vis_config.img_feature_type = ""
        vis_config.vl_layers = 4
        vis_config.la_layers = 9

        visual_model = model_class.from_pretrained(model_name_or_path, config=vis_config)

    return visual_model

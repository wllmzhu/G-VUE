# Recurrent VLN-BERT, 2020, by Yicong.Hong@anu.edu.au

from transformers import RobertaConfig, RobertaTokenizer, BertConfig, BertTokenizer

def get_tokenizer(cfg):
    if cfg.model.decoder.type == 'prevalent':
        # [ABLATION]
        if cfg.model.ablation.l_backbone == 'roberta':
            tokenizer_class = RobertaTokenizer
            tokenizer = tokenizer_class.from_pretrained(cfg.model.l_backbone.cfg_dir)
        elif cfg.model.ablation.l_backbone == 'lxmert':
            tokenizer_class = BertTokenizer
            tokenizer = tokenizer_class.from_pretrained('bert-base-uncased')
        else:
            print('l_backbone must be roberta or lxmert, please check the yaml file for model.ablation.l_backbone')
            exit(0)
        # ==========
    return tokenizer

def get_vlnbert_models(cfg, config=None):
    if cfg.model.decoder.type == 'prevalent':
        from models.nav_decoder.vlnbert.vlnbert_PREVALENT import VLNBert
        model_class = VLNBert
        model_name_or_path = cfg.train.pretrained_path.prevalent

        # [ABLATION]
        if cfg.model.ablation.l_backbone == 'roberta':
            vis_config = RobertaConfig.from_pretrained(cfg.model.l_backbone.cfg_dir)
        elif cfg.model.ablation.l_backbone == 'lxmert':
            vis_config = BertConfig.from_pretrained('bert-base-uncased')
        else:
            print('l_backbone must be roberta or lxmert, please check the yaml file for model.ablation.l_backbone')
            exit(0)
        # ==========

        vis_config.img_feature_dim = cfg.model.v_sizes[-1] + 128
        vis_config.img_feature_type = ""
        vis_config.vl_layers = 4
        vis_config.la_layers = 9

        # [ABLATION]
        # If no need to make decoder from scratch (for both roberta and lxmert)
        if not cfg.model.ablation.decoder_fromscratch: 
            visual_model = model_class.from_pretrained(model_name_or_path, config=vis_config, cfg=cfg)
            
        # Otherwise, need to make decoder from scratch
        if cfg.model.ablation.l_backbone == 'roberta': # Simply initialize an empty VLNBert, the l_backbone will be loaded with RoBERTa weights.
            visual_model = model_class(config=vis_config, cfg=cfg)
        else: # LXMERT. We initialize an empty VLNBert, then load only the PREVALENT LXMERT weights into the corresponding fields in empty VLNBert.
            pretrained = model_class.from_pretrained(model_name_or_path, config=vis_config, cfg=cfg)
            visual_model = model_class(config=vis_config, cfg=cfg)
            want_list = ['embeddings.word_embeddings', 'embeddings.position_embeddings',
                            'embeddings.token_type_embeddings', 'embeddings.LayerNorm', 
                            'pooler.dense', 'lalayer.0', 'lalayer.1', 'lalayer.2', 
                            'lalayer.3', 'lalayer.4', 'lalayer.5', 'lalayer.6', 
                            'lalayer.7', 'lalayer.8',]
            to_be_loaded = visual_model.state_dict()
            for name, param in pretrained.named_parameters():
                for want in want_list:
                    if want in name:
                        to_be_loaded[name] = param
                        break
            visual_model.load_state_dict(to_be_loaded, strict=False)
        # ==========

    return visual_model

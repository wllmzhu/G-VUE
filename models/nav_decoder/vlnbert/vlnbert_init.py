# Recurrent VLN-BERT, 2020, by Yicong.Hong@anu.edu.au

from transformers import RobertaConfig, RobertaTokenizer, BertTokenizer, BertConfig

def get_tokenizer(cfg):
    if cfg.model.decoder.type == 'prevalent':

        # tokenizer_class = RobertaTokenizer
        # tokenizer = tokenizer_class.from_pretrained(cfg.model.l_backbone.cfg_dir)
        #==========
        tokenizer_class = BertTokenizer
        tokenizer = tokenizer_class.from_pretrained('bert-base-uncased')

    return tokenizer

def get_vlnbert_models(cfg, config=None):

    # config_class = RobertaConfig
    #===============
    config_class = BertConfig

    if cfg.model.decoder.type == 'prevalent':
        from models.nav_decoder.vlnbert.vlnbert_PREVALENT import VLNBert
        model_class = VLNBert
        model_name_or_path = cfg.train.pretrained_path.prevalent

        # vis_config = config_class.from_pretrained(cfg.model.l_backbone.cfg_dir)
        #===============
        vis_config = config_class.from_pretrained('bert-base-uncased')

        vis_config.img_feature_dim = cfg.model.v_sizes[-1] + 128
        vis_config.img_feature_type = ""
        vis_config.vl_layers = 4
        vis_config.la_layers = 9

        #========pretrained switch========
        temp = model_class.from_pretrained(model_name_or_path, config=vis_config, cfg=cfg)
        visual_model = model_class(config=vis_config, cfg=cfg)

        #want only the language encoder
        wanted_list = ['embeddings.word_embeddings', 'embeddings.position_embeddings',
                        'embeddings.token_type_embeddings', 'embeddings.LayerNorm', 
                        'pooler.dense', 'lalayer.0', 'lalayer.1', 'lalayer.2', 
                        'lalayer.3', 'lalayer.4', 'lalayer.5', 'lalayer.6', 
                        'lalayer.7', 'lalayer.8']

        state = visual_model.state_dict()
        pretrained_states = temp.state_dict()
        for name, _ in temp.named_parameters():
            if name in wanted_list:
                name1 = name + '.weight'
                name2 = name + '.bias'
                state[name1] = pretrained_states[name1]
                state[name2] = pretrained_states[name2]
        visual_model.load_state_dict(state, strict=False)

        #===================================================
        #visual_model = model_class(config=vis_config, cfg=cfg)

    return visual_model

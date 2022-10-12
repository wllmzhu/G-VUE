# Models

This directory contains the encoder-decoder model implementation.

`base.py`

* Contains the `JointModel` class, the central piece that combines vision backbone, language backbone, decoder, and loss. 


`v_backbone.py`

* Contains the seven vision backbones evaluated in the paper, include `ResNet_ImageNet`, `ResNet_MoCov3`, `ResNet_Ego4D`, `ResNet_CLIP`, `ViT_CLIP_32`, `ViT_CLIP_16`, `ViT_MAE`. Also contains a `Customized` placeholder, where the user can implement their own vision backbone.


`l_backbone.py`

* Contains the language backbones `RoBERTa` and `RoBERTa_R2R`, both derived from HuggingFace pretrained weights `roberta_base`. 


`decoder.py`

* Constains the `DenseType`, `LabelType`, and `Rec3DType` decoder.


`manipulation/`

* Directory contains the Manipulation/CLIPort decoder, `GVUEManipAgent`, and related modules (agent, environment, etc.).


`navigation/` 

* Directory contains the Navigation/R2R decoder, `GVUENavAgent`, and related modules (agent, environment, etc.).


`rec3d_decoder/` 

* Directory contains the 3D decoder, `Rec3DType`'s related modules (VQ-VAE, etc.). The decoder itself is implemented in `decoder.py`.


`loss.py`

* Contains implementations for task-specific or output-specific losses, packaged in the `LOSS` registry.


`metrics.py`

* Contains implementations for task-specific or output-specific metrics, packaged in the `METRICS` registry.


`decoder_utils.py`

* Utils for `decoder.py`.


`positional_embedding.py`

* Utils for `decoder.py`.


`transformer.py`

* Utils for `decoder.py` and `decoder_utils.py`


`ablation.py` 

* An ablation study on the VL-Retrieval task.


from .utils import preprocess
from .base import OneStreamAttentionLangFusion, OneStreamTransportLangFusion, TwoStreamClipLingUNetTransporterAgent


class GVUEAgent(TwoStreamClipLingUNetTransporterAgent):

    def __init__(self, name, cfg, train_ds, test_ds):
        self.stream_fcn = cfg.backbone.key
        super().__init__(name, cfg, train_ds, test_ds)

    def _build_model(self):
        self.in_shape = (224, 224, 6)
        self.attention = OneStreamAttentionLangFusion(
            stream_fcn=(self.stream_fcn, None),
            in_shape=self.in_shape,
            n_rotations=1,
            preprocess=preprocess,
            cfg=self.cfg,
            device=self.device_type,
        )
        self.transport = OneStreamTransportLangFusion(
            stream_fcn=(self.stream_fcn, None),
            in_shape=self.in_shape,
            n_rotations=self.n_rotations,
            crop_size=self.crop_size,
            preprocess=preprocess,
            cfg=self.cfg,
            device=self.device_type,
        )

from .utils import preprocess
from .base import OneStreamAttentionLangFusion, OneStreamTransportLangFusion, TwoStreamClipLingUNetTransporterAgent


class GVUEManipAgent(TwoStreamClipLingUNetTransporterAgent):

    def __init__(self, name, cfg, train_ds, test_ds):
        self.stream_fcn = cfg.backbone.key
        super().__init__(name, cfg, train_ds, test_ds)

    def _build_model(self):
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

class ClipLingUNetTransporterAgent(TwoStreamClipLingUNetTransporterAgent):
    # clip-only baseline in cliport paper
    def __init__(self, name, cfg, train_ds, test_ds):
        super().__init__(name, cfg, train_ds, test_ds)

    def _build_model(self):
        stream_fcn = 'clip_lingunet'
        self.attention = OneStreamAttentionLangFusion(
            stream_fcn=(stream_fcn, None),
            in_shape=self.in_shape,
            n_rotations=1,
            preprocess=preprocess,
            cfg=self.cfg,
            device=self.device_type,
        )
        self.transport = OneStreamTransportLangFusion(
            stream_fcn=(stream_fcn, None),
            in_shape=self.in_shape,
            n_rotations=self.n_rotations,
            crop_size=self.crop_size,
            preprocess=preprocess,
            cfg=self.cfg,
            device=self.device_type,
        )

from easydict import EasyDict as edict

cfg = edict()


# ================================================================
# Model Configuration : Backbone, Encoder, Decoder
# ================================================================

cfg.MODEL = edict()

# ================================================================
# Backbone Configuration : TYPE,
# ================================================================
cfg.MODEL.BACKBONE = edict()
cfg.MODEL.BACKBONE.TYPE = "intern_image"
cfg.MODEL.BACKBONE.PRETRAINED = ''
cfg.MODEL.BACKBONE.RESUME = ''

cfg.MODEL.BACKBONE.PARAMS = edict()
cfg.MODEL.BACKBONE.PARAMS.CORE_OP = 'DCNv3'
cfg.MODEL.BACKBONE.PARAMS.CHANNELS = 64
cfg.MODEL.BACKBONE.PARAMS.DEPTHS = [4, 4, 18, 4]
cfg.MODEL.BACKBONE.PARAMS.GROUPS = [4, 8, 16, 32]
cfg.MODEL.BACKBONE.PARAMS.NUM_CLASSES = 1000
cfg.MODEL.BACKBONE.PARAMS.MLP_RATIO = 4.0
cfg.MODEL.BACKBONE.PARAMS.DROP_RATE = 0.0
cfg.MODEL.BACKBONE.PARAMS.DROP_PATH_RATE = 0.1
cfg.MODEL.BACKBONE.PARAMS.DROP_PATH_TYPE = 'linear'
cfg.MODEL.BACKBONE.PARAMS.LAYER_SCALE = None
cfg.MODEL.BACKBONE.PARAMS.OFFSET_SCALE = 1.0
cfg.MODEL.BACKBONE.PARAMS.POST_NORM = False
cfg.MODEL.BACKBONE.PARAMS.DW_KERNEL_SIZE = None
cfg.MODEL.BACKBONE.PARAMS.USE_CLIP_PROJECTOR = False
cfg.MODEL.BACKBONE.PARAMS.LEVEL2_POST_NORM = False
cfg.MODEL.BACKBONE.PARAMS.LEVEL2_POST_NORM_BLOCK_IDS = None
cfg.MODEL.BACKBONE.PARAMS.RES_POST_NORM = False
cfg.MODEL.BACKBONE.PARAMS.CENTER_FEATURE_SCALE = False
cfg.MODEL.BACKBONE.PARAMS.REMOVE_CENTER = False
from dataclasses import dataclass, replace, field



# ================================================================
# Encoder Configuration :
# ================================================================

from copy import deepcopy as dc
from .inter_image import InternImage
import importlib

def build_intern_image(config):
    cfg = dc(config)
    model_type = cfg.TYPE
    params = cfg.PARAMS
    if model_type == 'intern_image':
        model = InternImage(
            core_op = params.CORE_OP,
            num_classes= params.NUM_CLASSES,
            channels = params.CHANNELS,
            depths = params.DEPTHS,
            groups = params.GROUPS,
            layer_scale= params.LAYER_SCALE,
            offset_scale = params.OFFSET_SCALE,
            post_norm = params.POST_NORM,
            mlp_ratio = params.MLP_RATIO,
            drop_path_rate = params.DROP_PATH_RATE,
            res_post_norm= params.RES_POST_NORM,
            dw_kernel_size = params.DW_KERNEL_SIZE,
            use_clip_projector = params.USE_CLIP_PROJECTOR,
            level2_post_norm = params.LEVEL2_POST_NORM,
            level2_post_norm_block_ids= params.LEVEL2_POST_NORM_BLOCK_IDS,
            center_feature_scale= params.CENTER_FEATURE_SCALE,
            remove_center= params.REMOVE_CENTER

        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    return model

from lib.utils.config import cfg
from lib.models.backbone.internimage import build_intern_image

if __name__ == "__main__":
    model = build_intern_image(cfg.MODEL.BACKBONE)
    print(model)
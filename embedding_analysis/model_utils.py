import ml_collections
from pathlib import Path
from src.configs import owl_v2_clip_b16, owl_v2_clip_l14
from src.models import TextZeroShotDetectionModule

def get_model_config(model_config_str: str, weight_base_path: Path) -> ml_collections.ConfigDict:
    config_mapping = {
        "owl_v2_clip_b16": owl_v2_clip_b16.get_config,
        "owl_v2_clip_l14": owl_v2_clip_l14.get_config,
    }

    if model_config_str not in config_mapping:
        raise ValueError(f"Invalid model config: {model_config_str}")

    config = config_mapping[model_config_str](weight_base_path)
    config.model.body.patch_size = int(config.model.body.variant[-2:])
    config.model.body.native_image_grid_size = config.dataset_configs.input_size // config.model.body.patch_size
    return config

def get_model():
    config = get_model_config("owl_v2_clip_b16", Path("/home/ubuntu/airis/ai/object_detection/assets/owlvit/weights"))
    model = TextZeroShotDetectionModule(
        body_configs=config.model.body,
        objectness_head_configs=config.model.objectness_head,
        normalize=config.model.normalize,
        box_bias=config.model.box_bias,
    )
    variables = model.load_variables(config.init_from.checkpoint_path)
    return model, variables

def create_jitted_functions(model, variables):
    import functools
    import jax

    image_embedder = jax.jit(functools.partial(model.apply, variables, train=False, method=model.image_embedder))
    objectness_predictor = jax.jit(functools.partial(model.apply, variables, method=model.objectness_predictor))
    box_predictor = jax.jit(functools.partial(model.apply, variables, method=model.box_predictor))
    return image_embedder, objectness_predictor, box_predictor
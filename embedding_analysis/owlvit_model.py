import numpy as np
import jax.numpy as jnp
from scipy.special import expit as sigmoid
from embedding_analysis.image_utils import normalize_vectors
import ml_collections
from pathlib import Path
from src.configs import owl_v2_clip_b16, owl_v2_clip_l14
from src.models import TextZeroShotDetectionModule

class OwlVit:
    def __init__(self, model_config_str: str, weight_base_path: Path):
        self.config = self.get_model_config(model_config_str, weight_base_path)
        self.model, self.variables = self.get_model()
        self.image_embedder, self.objectness_predictor, self.box_predictor = self.create_jitted_functions()

    def get_model_config(self, model_config_str: str, weight_base_path: Path) -> ml_collections.ConfigDict:
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

    def get_model(self):
        model = TextZeroShotDetectionModule(
            body_configs=self.config.model.body,
            objectness_head_configs=self.config.model.objectness_head,
            normalize=self.config.model.normalize,
            box_bias=self.config.model.box_bias,
        )
        variables = model.load_variables(self.config.init_from.checkpoint_path)
        return model, variables

    def create_jitted_functions(self):
        import functools
        import jax

        image_embedder = jax.jit(functools.partial(self.model.apply, self.variables, train=False, method=self.model.image_embedder))
        objectness_predictor = jax.jit(functools.partial(self.model.apply, self.variables, method=self.model.objectness_predictor))
        box_predictor = jax.jit(functools.partial(self.model.apply, self.variables, method=self.model.box_predictor))
        return image_embedder, objectness_predictor, box_predictor

    def embed_image(self, source_image):
        feature_map = self.image_embedder(source_image[None, ...])
        b, h, w, d = feature_map.shape
        image_features = feature_map.reshape(b, h * w, d)
        return feature_map, image_features

    def predict_objectness(self, image_features):
        objectnesses = self.objectness_predictor(image_features)["objectness_logits"]
        return sigmoid(np.array(objectnesses[0]))

    def predict_boxes(self, image_features, feature_map):
        source_boxes = self.box_predictor(image_features=image_features, feature_map=feature_map)["pred_boxes"]
        return np.array(source_boxes[0])

    def generate_image_embeddings(self, source_image):
        feature_map, image_features = self.embed_image(source_image)
        objectnesses = self.predict_objectness(image_features)
        source_boxes = self.predict_boxes(image_features, feature_map)
        normalized_features = normalize_vectors(np.array(image_features[0]))
        return normalized_features, objectnesses, source_boxes
import numpy as np
import jax.numpy as jnp
from scipy.special import expit as sigmoid
from embedding_analysis.image_utils import normalize_vectors


def embed_image(source_image, image_embedder):
    feature_map = image_embedder(source_image[None, ...])
    b, h, w, d = feature_map.shape
    image_features = feature_map.reshape(b, h * w, d)
    return feature_map, image_features


def predict_objectness(image_features, objectness_predictor):
    objectnesses = objectness_predictor(image_features)["objectness_logits"]
    return sigmoid(np.array(objectnesses[0]))


def predict_boxes(image_features, feature_map, box_predictor):
    source_boxes = box_predictor(image_features=image_features, feature_map=feature_map)["pred_boxes"]
    return np.array(source_boxes[0])


def generate_image_embeddings(source_image, image_embedder, objectness_predictor, box_predictor):
    feature_map, image_features = embed_image(source_image, image_embedder)
    objectnesses = predict_objectness(image_features, objectness_predictor)
    source_boxes = predict_boxes(image_features, feature_map, box_predictor)
    normalized_features = normalize_vectors(np.array(image_features[0]))
    return normalized_features, objectnesses, source_boxes

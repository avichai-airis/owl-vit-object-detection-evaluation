import os
import sys

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(root_dir)
import jax.numpy as jnp
from skimage import io as skimage_io
from src.preprocessing import prepare_image_queries, closest_divisible_size
import numpy as np


def read_image(image_path: str, divide_by: int, resize_dim: int):
    original_image = skimage_io.imread(image_path)[:, :, :3]
    resize_image = prepare_image_queries(
        original_image, closest_divisible_size(max(original_image.shape[:2]), divide_by, resize_dim)
    )
    query_image = jnp.array(resize_image)
    return original_image, resize_image, query_image


def normalize_vectors(image_features):
    norms = np.linalg.norm(image_features, axis=1, keepdims=True)
    norms[norms == 0] = 1
    normalized_image_features = image_features / norms
    return normalized_image_features

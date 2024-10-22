import functools
import os.path
from pathlib import Path
from typing import Dict

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import ml_collections
import numpy as np
from scipy.special import expit as sigmoid
from skimage import io as skimage_io
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from src.configs import owl_v2_clip_b16, owl_v2_clip_l14
from src.models import TextZeroShotDetectionModule
from src.preprocessing import prepare_image_queries, closest_divisible_size
from sklearn.cluster import DBSCAN
from scipy.stats import gaussian_kde
from tqdm import tqdm


def normalize_vectors(image_features):
    # Compute the L2 norm for each vector
    norms = np.linalg.norm(image_features, axis=1, keepdims=True)
    # Avoid division by zero
    norms[norms == 0] = 1
    # Normalize each vector
    normalized_image_features = image_features / norms
    return normalized_image_features


def read_image(image_path: str, divide_by: int, resize_dim: int):
    original_image = skimage_io.imread(image_path)[:, :, :3]
    resize_image = prepare_image_queries(original_image,
                                         closest_divisible_size(max(original_image.shape[:2]), divide_by,
                                                                resize_dim))
    query_image = jnp.array(resize_image)
    return original_image, resize_image, query_image


def get_model_config(model_config_str: str, weight_base_path: Path) -> ml_collections.ConfigDict:
    """
    Get model config from the model config string provided.
    Args:
        model_config_str: str, model config string. for now, it can be one of the following:
            - "owl_v2_clip_b16"
            - "owl_v2_clip_l14"
        weight_base_path: Path, base path for the model weights
    Returns: ml_collections.ConfigDict, model config object

    """
    config_mapping = {
        "owl_v2_clip_b16": owl_v2_clip_b16.get_config,
        "owl_v2_clip_l14": owl_v2_clip_l14.get_config,
    }

    if model_config_str not in config_mapping:
        raise ValueError(f"Invalid model config: {model_config_str}")

    return config_mapping[model_config_str](weight_base_path)


def get_model():
    """
    Load the model and variables using the given config.
    Args:
        config: model config object containing the model weights path and other model related configs

    Returns: tuple, (model, variables)

    """
    config = get_model_config("owl_v2_clip_b16", Path("/home/ubuntu/airis/ai/object_detection/assets/owlvit/weights"))
    config.model.body.patch_size = int(config.model.body.variant[-2:])
    config.model.body.native_image_grid_size = config.dataset_configs.input_size // config.model.body.patch_size

    # Initialize the model
    model = TextZeroShotDetectionModule(
        body_configs=config.model.body,
        objectness_head_configs=config.model.objectness_head,
        normalize=config.model.normalize,
        box_bias=config.model.box_bias,
    )
    # Load the model weights
    variables = model.load_variables(config.init_from.checkpoint_path)

    return model, variables


def create_jitted_functions(model, variables):
    image_embedder = jax.jit(
        functools.partial(
            model.apply, variables, train=False, method=model.image_embedder
        )
    )

    objectness_predictor = jax.jit(
        functools.partial(
            model.apply, variables, method=model.objectness_predictor
        )
    )

    box_predictor = jax.jit(
        functools.partial(model.apply, variables, method=model.box_predictor)
    )

    return image_embedder, objectness_predictor, box_predictor


def embed_image(source_image, image_embedder):
    feature_map = image_embedder(source_image[None, ...])
    b, h, w, d = feature_map.shape
    image_features = feature_map.reshape(b, h * w, d)
    return feature_map, image_features


def predict_objectness(image_features, objectness_predictor):
    objectnesses = objectness_predictor(image_features)['objectness_logits']
    return sigmoid(np.array(objectnesses[0]))


def predict_boxes(image_features, feature_map, box_predictor):
    source_boxes = box_predictor(
        image_features=image_features, feature_map=feature_map
    )['pred_boxes']
    return np.array(source_boxes[0])


def generate_image_embeddings(source_image, image_embedder, objectness_predictor, box_predictor):
    feature_map, image_features = embed_image(source_image, image_embedder)
    objectnesses = predict_objectness(image_features, objectness_predictor)
    source_boxes = predict_boxes(image_features, feature_map, box_predictor)
    return normalize_vectors(np.array(image_features[0])), objectnesses, source_boxes



# Assuming objectnesses is a numpy array with 3600 scores
def plot_objectness_distribution(objectnesses, save_path=None):
    plt.figure(figsize=(10, 6))
    plt.hist(objectnesses, bins=50, color='blue', alpha=0.7)
    plt.xlabel('Objectness Score')
    plt.ylabel('Frequency')
    plt.title('Distribution of Objectness Scores')
    plt.grid(True)
    # log scale
    plt.yscale('log')
    # save the plot
    if save_path:
        plt.savefig(os.path.join(save_path, 'objectness_distribution.png'))
    else:
        plt.show()


def visualize_image_features_kmeans(image_features, n_clusters=5, save_path=None):
    # Reduce dimensionality to 2D using t-SNE
    tsne = TSNE(n_components=2, random_state=42)
    image_features_2d = tsne.fit_transform(image_features)

    # Apply K-Means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(image_features_2d)

    # Plot the clustered data
    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(image_features_2d[:, 0], image_features_2d[:, 1], c=clusters, cmap='viridis', alpha=0.7)
    plt.colorbar(scatter, label='Cluster')
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    plt.title('t-SNE Visualization of Image Features with K-Means Clustering')
    plt.grid(True)

    # Save or show the plot
    if save_path:
        plt.savefig(os.path.join(save_path, 'image_features_clustering_kmeans.png'))
    else:
        plt.show()
    plt.close()


def visualize_image_features_dbscan(image_features, eps=0.25, min_samples=10, save_path=None):
    # Apply DBSCAN clustering in the original vector space
    dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric='cosine')
    clusters = dbscan.fit_predict(image_features)

    # Reduce dimensionality to 2D using t-SNE
    tsne = TSNE(n_components=2, random_state=42)
    image_features_2d = tsne.fit_transform(image_features)

    # Plot the clustered data
    plt.figure(figsize=(10, 6))
    unique_clusters = np.unique(clusters)
    for cluster in unique_clusters:
        cluster_points = image_features_2d[clusters == cluster]
        plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f'Cluster {cluster}', alpha=0.7)

    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.title('t-SNE Visualization of Image Features with DBSCAN Clustering')
    plt.grid(True)
    plt.tight_layout()

    # Save or show the plot
    if save_path:
        plt.savefig(os.path.join(save_path, 'image_features_clustering_dbscan.png'))
    else:
        plt.show()
    plt.close()


def plot_bb_on_image(source_image,source_boxes, objectnesses, objectness_threshold, save_path):
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    ax.imshow(source_image, extent=(0, 1, 1, 0))
    ax.set_axis_off()

    for i, (box, objectness) in enumerate(zip(source_boxes, objectnesses)):
        if objectness < objectness_threshold:
            continue

        cx, cy, w, h = box
        ax.plot(
            [cx - w / 2, cx + w / 2, cx + w / 2, cx - w / 2, cx - w / 2],
            [cy - h / 2, cy - h / 2, cy + h / 2, cy + h / 2, cy - h / 2],
            color='lime',
        )

        ax.text(
            cx - w / 2 + 0.015,
            cy + h / 2 - 0.015,
            f'objectness: {objectness:1.2f}',
            ha='left',
            va='bottom',
            color='black',
            bbox={
                'facecolor': 'white',
                'edgecolor': 'lime',
                'boxstyle': 'square,pad=.3',
            },
        )

    ax.set_xlim(0, 1)
    ax.set_ylim(1, 0)
    # save the plot
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'bounding_boxes.png'))
    plt.close(fig)


def plot_correlation_objectness_bbox_size(objectnesses, source_boxes, save_path=None):
    # Calculate bounding box sizes (area)
    bbox_sizes = [(w * h) for _, _, w, h in source_boxes]

    # Calculate the density of each point
    xy = np.vstack([bbox_sizes, objectnesses])
    z = gaussian_kde(xy)(xy)

    # Create scatter plot with density coloring
    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(bbox_sizes, objectnesses, c=z, cmap='viridis', alpha=0.7)
    plt.colorbar(scatter, label='Density')
    plt.xlabel('Bounding Box Size (Area)')
    plt.ylabel('Objectness Score')
    plt.title('Correlation between Objectness Scores and Bounding Box Sizes with Density Coloring')
    plt.grid(True)

    # Save or show the plot
    if save_path:
        plt.savefig(os.path.join(save_path, 'correlation_objectness_bbox_size_density.png'))
    else:
        plt.show()
    plt.close()


def plot_correlation_objectness_bbox_size_hexbin(objectnesses, source_boxes, save_path=None):
    # Calculate bounding box sizes (area)
    bbox_sizes = [(w * h) for _, _, w, h in source_boxes]

    # Create hexbin plot
    plt.figure(figsize=(10, 6))
    hb = plt.hexbin(bbox_sizes, objectnesses, gridsize=50, cmap='viridis', mincnt=1)
    plt.colorbar(hb, label='Count')
    plt.xlabel('Bounding Box Size (Area)')
    plt.ylabel('Objectness Score')
    plt.title('Correlation between Objectness Scores and Bounding Box Sizes with Hexbin Plot')
    plt.grid(True)

    # Save or show the plot
    if save_path:
        plt.savefig(os.path.join(save_path, 'correlation_objectness_bbox_size_hexbin.png'))
    else:
        plt.show()
    plt.close()


def dbscan_clustering_example(save_path=None):
    from sklearn.datasets import make_blobs
    from sklearn.preprocessing import StandardScaler
    # Generate synthetic data
    n_samples = 1500
    random_state = 170
    X, _ = make_blobs(n_samples=n_samples, random_state=random_state)

    # Standardize features by removing the mean and scaling to unit variance
    X = StandardScaler().fit_transform(X)

    # Apply DBSCAN
    eps = 0.25
    min_samples = 10
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    clusters = dbscan.fit_predict(X)

    # Plot the clustered data
    plt.figure(figsize=(10, 6))
    unique_clusters = np.unique(clusters)
    for cluster in unique_clusters:
        cluster_points = X[clusters == cluster]
        plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f'Cluster {cluster}', alpha=0.7)

    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('DBSCAN Clustering on Synthetic Data')
    plt.legend()
    plt.grid(True)
    if save_path:
        plt.savefig(os.path.join(save_path, 'dbscan_clustering_example.png'))
    else:
        plt.show()
    plt.close()


def process_image(image_path, image_embedder, objectness_predictor, box_predictor, objectness_threshold):
    source_image, resize_image, query_image = read_image(image_path, 16, 960)
    image_features, objectnesses, source_boxes = generate_image_embeddings(query_image, image_embedder,
                                                                           objectness_predictor, box_predictor)

    # Filter out vectors with objectness score smaller than the threshold
    filtered_features = image_features[objectnesses > objectness_threshold]
    return filtered_features


def process_images_in_folder(folder_path, model, variables, objectness_threshold=0.1, run_in_parallel=True):
    from concurrent.futures import ThreadPoolExecutor
    # get jittered functions
    image_embedder, objectness_predictor, box_predictor = create_jitted_functions(model, variables)

    image_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if
                   f.endswith(".png") or f.endswith(".jpg")]
    if run_in_parallel:
        with ThreadPoolExecutor() as executor:
            results = list(tqdm(executor.map(
                lambda f: process_image(f, image_embedder, objectness_predictor, box_predictor, objectness_threshold),
                image_files), total=len(image_files)))
    else:
        results = []
        for f in tqdm(image_files):
            results.append(process_image(f, image_embedder, objectness_predictor, box_predictor, objectness_threshold))

    return np.vstack(results)


def main():
    objectness_threshold = 0.1
    save_path = "/home/ubuntu/Data/video_for_debug_sampling/dog_example/sampled_images"
    model, variables = get_model()
    image_path = "/home/ubuntu/Data/video_for_debug_sampling/dog_example/sampled_images/dog.png"
    source_image, resize_image, query_image = read_image(image_path, 16, 960)
    # dbscan_clustering_example(save_path)
    image_features, objectnesses, source_boxes = generate_image_embeddings(query_image, model, variables)

    plot_bb_on_image(resize_image,source_boxes, objectnesses, objectness_threshold, save_path)
    print("image_features:", image_features.shape)
    print("objectnesses:", objectnesses.shape)
    print("source_boxes:", source_boxes.shape)
    plot_correlation_objectness_bbox_size(objectnesses, source_boxes, save_path=save_path)
    # take only the image_features  that the objectness score is higher than the threshold
    # image_features = image_features[objectnesses > objectness_threshold]
    print(f"image_features: {image_features.shape}")
    plot_objectness_distribution(objectnesses, save_path)
    visualize_image_features_kmeans(image_features, 5, save_path)
    visualize_image_features_dbscan(image_features, 0.01, 10, save_path)
    plot_correlation_objectness_bbox_size_hexbin(objectnesses, source_boxes, save_path)


if __name__ == '__main__':
    objectness_threshold = 0.1
    save_npy_path = os.path.join('/home/ubuntu/Data/video_for_debug_sampling/object_video_5_vectors/vectors',
                                 'all_image_features_test.npy')
    save_path = "/home/ubuntu/Data/video_for_debug_sampling/object_video_5_vectors/plots"
    # check if the file exists
    if os.path.exists(save_npy_path):
        all_image_features = np.load(save_npy_path)

    else:
        folder_path = "/home/ubuntu/Data/video_for_debug_sampling/object_video_5/sampled_images"
        model, variables = get_model()
        image_embedder, objectness_predictor, box_predictor = create_jitted_functions(model, variables)
        all_image_features = process_images_in_folder(folder_path, model, variables, objectness_threshold, run_in_parallel=True)
        # save the image features
        np.save(save_npy_path, all_image_features)

    visualize_image_features_dbscan(all_image_features, eps=0.01, min_samples=5, save_path=save_path)

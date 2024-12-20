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
import umap
import time
import hdbscan
from src.postprocessing import box_cxcywh_to_xyxy


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
        plt.savefig(
            os.path.join(save_path, f'eps_{eps}_min_samples_{min_samples}_image_features_clustering_dbscan.png'))
    else:
        plt.show()
    plt.close()


def plot_bb_on_image(source_image, source_boxes, objectnesses, objectness_threshold, image_name, save_path):
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
    plt.savefig(os.path.join(save_path, f'{image_name}_bounding_boxes.png'))
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
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True)
    if save_path:
        plt.savefig(os.path.join(save_path, 'dbscan_clustering_example.png'))
    else:
        plt.show()
    plt.close()


def process_image(image_path, image_embedder, objectness_predictor, box_predictor, objectness_threshold, save_path=None):
    source_image, resize_image, query_image = read_image(image_path, 16, 960)
    image_features, objectnesses, source_boxes = generate_image_embeddings(query_image, image_embedder,
                                                                           objectness_predictor, box_predictor)

    # Filter out vectors with objectness score smaller than the threshold
    filtered_features = image_features[objectnesses > objectness_threshold]
    filtered_boxes = source_boxes[objectnesses > objectness_threshold]

    # Plot bounding box size histogram
    image_name = Path(image_path).stem
    plot_bbox_size_histogram(image_name, filtered_boxes, resize_image.shape, save_path)

    return filtered_features, filtered_boxes, resize_image, Path(image_path).stem


def process_images_in_folder(folder_path, model, variables, objectness_threshold=0.1, run_in_parallel=True, save_path=None):
    from concurrent.futures import ThreadPoolExecutor
    # get jittered functions
    image_embedder, objectness_predictor, box_predictor = create_jitted_functions(model, variables)

    image_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if
                   f.endswith(".png") or f.endswith(".jpg")]
    all_features = []
    all_boxes = []
    all_image_names = []
    feature_intervals = []

    if run_in_parallel:
        with ThreadPoolExecutor() as executor:
            results = list(tqdm(executor.map(
                lambda f: process_image(f, image_embedder, objectness_predictor, box_predictor, objectness_threshold, save_path),
                image_files), total=len(image_files)))
    else:
        results = []
        for f in tqdm(image_files):
            results.append(process_image(f, image_embedder, objectness_predictor, box_predictor, objectness_threshold, save_path))

    current_index = 0
    for features, boxes, image, image_name in results:
        all_features.append(features)
        all_boxes.append(boxes)
        all_image_names.append(image_name)
        feature_intervals.append((current_index, current_index + len(features)))
        current_index += len(features)

    return np.vstack(all_features), all_boxes, all_image_names, feature_intervals


def main():
    objectness_threshold = 0.1
    save_path = "/home/ubuntu/Data/video_for_debug_sampling/car_and_person/sampled_images"
    image_path = "/home/ubuntu/Data/video_for_debug_sampling/car_and_person/sampled_images/"
    model, variables = get_model()
    image_embedder, objectness_predictor, box_predictor = create_jitted_functions(model, variables)
    source_image, resize_image, query_image = read_image(image_path, 16, 960)
    # dbscan_clustering_example(save_path)
    image_features, objectnesses, source_boxes = generate_image_embeddings(query_image, image_embedder,
                                                                           objectness_predictor, box_predictor)

    plot_bb_on_image(resize_image, source_boxes, objectnesses, objectness_threshold, save_path)
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


def visualize_image_features_hdbscan(image_features, boxes, images_folder_path, image_names,feature_intervals, min_cluster_size=1, save_path=None,objectness_threshold=0.1):
    save_path = os.path.join(save_path, f'objectness_th_{str(objectness_threshold)}')
    os.makedirs(save_path, exist_ok=True)
    # Reduce dimensionality to 2D using UMAP
    umap_reducer = umap.UMAP(n_neighbors=15, n_components=2, metric='cosine', n_jobs=-1)
    image_features_2d = umap_reducer.fit_transform(image_features)

    # Apply HDBSCAN clustering
    hdbscan_clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size)
    clusters = hdbscan_clusterer.fit_predict(image_features_2d)

    # Plot the clustered data
    plt.figure(figsize=(26, 15))
    unique_clusters = np.unique(clusters)
    for cluster in unique_clusters:
        cluster_points = image_features_2d[clusters == cluster]
        plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f'Cluster {cluster}', alpha=0.7)
        if cluster != -1:
            # Calculate the centroid of the cluster
            centroid = cluster_points.mean(axis=0)

            # Calculate the bounding box of the cluster
            min_x, min_y = cluster_points.min(axis=0)
            max_x, max_y = cluster_points.max(axis=0)

            # Calculate the radius of the circle using the diagonal of the bounding box
            radius = np.sqrt((max_x - min_x) ** 2 + (max_y - min_y) ** 2) / 2

            # Draw the circle
            circle = plt.Circle(centroid, radius, color='red', fill=False, linestyle='--')
            plt.gca().add_patch(circle)

            # Add the cluster number outside the circle
            plt.text(centroid[0] + radius, centroid[1], str(cluster), fontsize=12, weight='bold', color='red')

    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', ncol=3)
    plt.title('UMAP Visualization of Image Features with HDBSCAN Clustering')
    plt.grid(True)
    plt.tight_layout()

    # Save or show the plot
    if save_path:
        plt.savefig(os.path.join(save_path, 'image_features_hdbscan.png'))
    else:
        plt.show()
    plt.close()

    # Plot bounding boxes with cluster numbers
    for image_name, image_boxes, feature_interval in zip(image_names, boxes, feature_intervals):
        image_path = os.path.join(images_folder_path, f'{image_name}.png')
        original_image = skimage_io.imread(image_path)[:, :, :3]
        height, width, _ = original_image.shape
        fig, ax = plt.subplots(1, 1, figsize=(width / 100, height / 100), dpi=100)
        ax.imshow(original_image)
        ax.set_axis_off()
        image_clusters = clusters[feature_interval[0]:feature_interval[1]]
        for box, cluster in zip(list(image_boxes), image_clusters):
            xtl, ytl, xbr, ybr = box_cxcywh_to_xyxy(box, (width, height))
            ax.plot(
                [xtl, xbr, xbr, xtl, xtl],
                [ytl, ytl, ybr, ybr, ytl],
                color='lime',
            )
            ax.text(
                xtl + 0.015 * width,
                ybr - 0.015 * height,
                f'Cluster: {cluster}',
                ha='left',
                va='bottom',
                color='black',
                bbox={
                    'facecolor': 'white',
                    'edgecolor': 'lime',
                    'boxstyle': 'square,pad=.3',
                },
            )

        ax.set_xlim(0, width)
        ax.set_ylim(height, 0)
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
        plt.savefig(os.path.join(save_path, f'{image_name}_bounding_boxes_with_clusters.png'), bbox_inches='tight', pad_inches=0)
        plt.close(fig)


def visualize_image_features_umap(image_features, eps, min_samples, save_path=None):
    print(f"Start UMAP dimensionality reduction...")
    start_time = time.time()
    # Reduce dimensionality to 2D using UMAP
    umap_reducer = umap.UMAP(n_neighbors=5, n_components=2, metric='cosine', n_jobs=-1)
    image_features_2d = umap_reducer.fit_transform(image_features)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"UMAP dimensionality reduction took {elapsed_time:.2f} seconds")
    # add dbscan clustering
    # Apply DBSCAN clustering in the original vector space
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    clusters = dbscan.fit_predict(image_features)
    # draw each cluster with different color
    plt.figure(figsize=(10, 6))
    unique_clusters = np.unique(clusters)
    for cluster in unique_clusters:
        cluster_points = image_features_2d[clusters == cluster]
        plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f'Cluster {cluster}', alpha=0.7)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    # # Plot the clustered data
    # plt.figure(figsize=(10, 6))
    # plt.scatter(image_features_2d[:, 0], image_features_2d[:, 1], alpha=0.7)

    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('Umap Visualization of Image Features')
    plt.grid(True)
    plt.tight_layout()

    # Save or show the plot
    if save_path:
        plt.savefig(os.path.join(save_path, f'eps_{eps}_min_samples_{min_samples}_image_features_umap.png'))
    else:
        plt.show()
    plt.close()
def plot_bbox_size_histogram(image_name, source_boxes, image_shape, save_path=None):
    # Calculate bounding box sizes (area) and normalize by image size
    bbox_sizes = [(w * h) / (image_shape[0] * image_shape[1]) for _, _, w, h in source_boxes]

    # Plot histogram
    plt.figure(figsize=(10, 6))
    plt.hist(bbox_sizes, bins=50, color='blue', alpha=0.7)
    plt.xlabel('Normalized Bounding Box Size')
    plt.ylabel('Frequency')
    plt.title(f'Histogram of Normalized Bounding Box Sizes for {image_name}')
    plt.grid(True)
    # log scale
    plt.yscale('log')
    # Save or show the plot
    if save_path:
        plt.savefig(os.path.join(save_path, f'{image_name}_bbox_size_histogram.png'))
    else:
        plt.show()
    plt.close()

def visualize_detected_object_features(objectness_threshold=0.1,
                                       save_path='/home/ubuntu/Data/video_for_debug_sampling/object_video_5_vectors/plots',
                                       folder_path='/home/ubuntu/Data/video_for_debug_sampling/object_video_5/sampled_images'):
    os.makedirs(save_path, exist_ok=True)

    model, variables = get_model()
    image_features, boxes, image_names,feature_intervals = process_images_in_folder(folder_path, model, variables, objectness_threshold,
                                                  run_in_parallel=True, save_path=save_path)


    visualize_image_features_hdbscan(image_features, boxes,folder_path ,image_names,feature_intervals, min_cluster_size=2, save_path=save_path, objectness_threshold=objectness_threshold)


def process_and_plot_bb_on_images(objectness_thresholds, images_path, save_base_path):
    model, variables = get_model()
    image_embedder, objectness_predictor, box_predictor = create_jitted_functions(model, variables)

    for objectness_threshold in objectness_thresholds:
        save_path = os.path.join(save_base_path, f"th_{objectness_threshold}_image_with_bb")
        os.makedirs(save_path, exist_ok=True)

        # Get all images in the folder
        image_files = [os.path.join(images_path, f) for f in os.listdir(images_path) if
                       f.endswith(".png") or f.endswith(".jpg")]

        for image_file in tqdm(image_files):
            source_image, resize_image, query_image = read_image(image_file, 16, 960)
            image_features, objectnesses, source_boxes = generate_image_embeddings(query_image, image_embedder,
                                                                                   objectness_predictor, box_predictor)

            # Get image name from the image path
            image_name = Path(image_file).stem
            plot_bb_on_image(resize_image, source_boxes, objectnesses, objectness_threshold, image_name, save_path)


if __name__ == '__main__':
    # visualize_detected_object_features()
    # process_and_plot_bb_on_images([0.2, 0.3, 0.4, 0.5],
    #                               "/home/ubuntu/Data/video_for_debug_sampling/car_and_person/sampled_images",
    #                               "/home/ubuntu/Data/video_for_debug_sampling/car_and_person/th_")
    visualize_detected_object_features(0.2,
                                       '/home/ubuntu/Data/video_for_debug_sampling/car_and_person/plots',
                                       '/home/ubuntu/Data/video_for_debug_sampling/car_and_person/sampled_images')
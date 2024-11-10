from image_utils import read_image
from model_utils import get_model, create_jitted_functions
from embedding_utils import generate_image_embeddings
from visualization_utils import (
    plot_objectness_distribution,
    plot_bb_on_image,
    plot_correlation_objectness_bbox_size,
    plot_bbox_size_histogram,
)
from clustering_utils import (
    visualize_image_features_kmeans,
    visualize_image_features_dbscan,
    visualize_image_features_hdbscan,
    visualize_image_features_umap,
)
import os
import sys
from pathlib import Path
import numpy as np
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(root_dir)


def process_image(image_path, image_embedder, objectness_predictor, box_predictor, objectness_threshold, save_path):
    _, resize_image, query_image = read_image(image_path, 16, 960)
    image_features, objectnesses, source_boxes = generate_image_embeddings(
        query_image, image_embedder, objectness_predictor, box_predictor
    )
    image_name = Path(image_path).stem
    plot_objectness_distribution(objectnesses, save_path, f"{image_name}_objectness_distribution.png")
    filtered_features = image_features[objectnesses > objectness_threshold]
    filtered_boxes = source_boxes[objectnesses > objectness_threshold]
    filtered_objectnesses = objectnesses[objectnesses > objectness_threshold]
    plot_bbox_size_histogram(image_name, filtered_boxes, resize_image.shape, save_path)
    plot_correlation_objectness_bbox_size(
        filtered_objectnesses, filtered_boxes, image_name, resize_image.shape, save_path
    )
    return filtered_features, filtered_boxes, resize_image, image_name


def process_images_in_folder(save_path, folder_path, model, variables, objectness_threshold=0.1, run_in_parallel=True):
    image_embedder, objectness_predictor, box_predictor = create_jitted_functions(model, variables)
    image_files = [
        os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith(".png") or f.endswith(".jpg")
    ]
    all_features = []
    all_boxes = []
    all_image_names = []
    feature_intervals = []
    if run_in_parallel:
        with ThreadPoolExecutor() as executor:
            results = list(
                tqdm(
                    executor.map(
                        lambda f: process_image(
                            f, image_embedder, objectness_predictor, box_predictor, objectness_threshold, save_path
                        ),
                        image_files,
                    ),
                    total=len(image_files),
                )
            )
    else:
        results = []
        for f in tqdm(image_files):
            results.append(
                process_image(f, image_embedder, objectness_predictor, box_predictor, objectness_threshold, save_path)
            )
    current_index = 0
    for features, boxes, image, image_name in results:
        all_features.append(features)
        all_boxes.append(boxes)
        all_image_names.append(image_name)
        feature_intervals.append((current_index, current_index + len(features)))
        current_index += len(features)
    return np.vstack(all_features), all_boxes, all_image_names, feature_intervals


def visualize_detected_object_features(
    objectness_threshold=0.1,
    save_path="/home/ubuntu/Data/video_for_debug_sampling/object_video_5_vectors/plots",
    folder_path="/home/ubuntu/Data/video_for_debug_sampling/object_video_5/sampled_images",
    run_in_parallel=True,
):
    save_path = os.path.join(save_path, f"objectness_th_{str(objectness_threshold)}")
    os.makedirs(save_path, exist_ok=True)
    model, variables = get_model()
    image_features, boxes, image_names, feature_intervals = process_images_in_folder(
        save_path, folder_path, model, variables, objectness_threshold, run_in_parallel=run_in_parallel
    )
    visualize_image_features_hdbscan(
        image_features,
        boxes,
        folder_path,
        image_names,
        feature_intervals,
        min_cluster_size=2,
        save_path=save_path,
    )


def main():
    visualize_detected_object_features(
        objectness_threshold=0.15,
        save_path="/home/ubuntu/Data/video_for_debug_sampling/car_and_person/plots",
        folder_path="/home/ubuntu/Data/video_for_debug_sampling/car_and_person/sampled_images",
        run_in_parallel=False,
    )


if __name__ == "__main__":
    main()

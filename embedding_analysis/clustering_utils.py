import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, DBSCAN
from sklearn.manifold import TSNE
import umap
import hdbscan
import os

def visualize_image_features_kmeans(image_features, n_clusters=5, save_path=None):
    tsne = TSNE(n_components=2, random_state=42)
    image_features_2d = tsne.fit_transform(image_features)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(image_features_2d)
    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(image_features_2d[:, 0], image_features_2d[:, 1], c=clusters, cmap="viridis", alpha=0.7)
    plt.colorbar(scatter, label="Cluster")
    plt.xlabel("t-SNE Component 1")
    plt.ylabel("t-SNE Component 2")
    plt.title("t-SNE Visualization of Image Features with K-Means Clustering")
    plt.grid(True)
    if save_path:
        plt.savefig(os.path.join(save_path, "image_features_clustering_kmeans.png"))
    else:
        plt.show()
    plt.close()

def visualize_image_features_dbscan(image_features, eps=0.25, min_samples=10, save_path=None):
    dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric="cosine")
    clusters = dbscan.fit_predict(image_features)
    tsne = TSNE(n_components=2, random_state=42)
    image_features_2d = tsne.fit_transform(image_features)
    plt.figure(figsize=(10, 6))
    unique_clusters = np.unique(clusters)
    for cluster in unique_clusters:
        cluster_points = image_features_2d[clusters == cluster]
        plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f"Cluster {cluster}", alpha=0.7)
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.title("t-SNE Visualization of Image Features with DBSCAN Clustering")
    plt.grid(True)
    plt.tight_layout()
    if save_path:
        plt.savefig(os.path.join(save_path, f"eps_{eps}_min_samples_{min_samples}_image_features_clustering_dbscan.png"))
    else:
        plt.show()
    plt.close()

def visualize_image_features_hdbscan(
    image_features,
    boxes,
    images_folder_path,
    image_names,
    feature_intervals,
    min_cluster_size=1,
    save_path=None,
):
    umap_reducer = umap.UMAP(n_neighbors=15, n_components=2, metric="cosine", n_jobs=-1)
    image_features_2d = umap_reducer.fit_transform(image_features)
    hdbscan_clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size)
    clusters = hdbscan_clusterer.fit_predict(image_features_2d)
    plt.figure(figsize=(26, 15))
    unique_clusters = np.unique(clusters)
    for cluster in unique_clusters:
        cluster_points = image_features_2d[clusters == cluster]
        plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f"Cluster {cluster}", alpha=0.7)
        if cluster != -1:
            centroid = cluster_points.mean(axis=0)
            min_x, min_y = cluster_points.min(axis=0)
            max_x, max_y = cluster_points.max(axis=0)
            radius = np.sqrt((max_x - min_x) ** 2 + (max_y - min_y) ** 2) / 2
            circle = plt.Circle(centroid, radius, color="red", fill=False, linestyle="--")
            plt.gca().add_patch(circle)
            plt.text(centroid[0] + radius, centroid[1], str(cluster), fontsize=12, weight="bold", color="red")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", ncol=3)
    plt.title("UMAP Visualization of Image Features with HDBSCAN Clustering")
    plt.grid(True)
    plt.tight_layout()
    if save_path:
        plt.savefig(os.path.join(save_path, "image_features_hdbscan.png"))
    else:
        plt.show()
    plt.close()

    from skimage import io as skimage_io
    from src.postprocessing import box_cxcywh_to_xyxy

    for image_name, image_boxes, feature_interval in zip(image_names, boxes, feature_intervals):
        image_path = os.path.join(images_folder_path, f"{image_name}.png")
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
                color="lime",
            )
            ax.text(
                xtl + 0.015 * width,
                ybr - 0.015 * height,
                f"Cluster: {cluster}",
                ha="left",
                va="bottom",
                color="black",
                bbox={
                    "facecolor": "white",
                    "edgecolor": "lime",
                    "boxstyle": "square,pad=.3",
                },
            )

        ax.set_xlim(0, width)
        ax.set_ylim(height, 0)
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
        plt.savefig(
            os.path.join(save_path, f"{image_name}_bounding_boxes_with_clusters.png"), bbox_inches="tight", pad_inches=0
        )
        plt.close(fig)

def visualize_image_features_umap(image_features, eps, min_samples, save_path=None):
    import time
    print(f"Start UMAP dimensionality reduction...")
    start_time = time.time()
    umap_reducer = umap.UMAP(n_neighbors=5, n_components=2, metric="cosine", n_jobs=-1)
    image_features_2d = umap_reducer.fit_transform(image_features)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"UMAP dimensionality reduction took {elapsed_time:.2f} seconds")
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    clusters = dbscan.fit_predict(image_features)
    plt.figure(figsize=(10, 6))
    unique_clusters = np.unique(clusters)
    for cluster in unique_clusters:
        cluster_points = image_features_2d[clusters == cluster]
        plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f"Cluster {cluster}", alpha=0.7)
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.title("UMAP Visualization of Image Features")
    plt.grid(True)
    plt.tight_layout()
    if save_path:
        plt.savefig(os.path.join(save_path, f"eps_{eps}_min_samples_{min_samples}_image_features_umap.png"))
    else:
        plt.show()
    plt.close()
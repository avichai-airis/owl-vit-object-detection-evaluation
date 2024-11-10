import matplotlib.pyplot as plt
import os
import numpy as np
from scipy.stats import gaussian_kde

def plot_objectness_distribution(objectnesses, save_path=None, save_file_name="objectness_distribution.png"):
    plt.figure(figsize=(10, 6))
    plt.hist(objectnesses, bins=50, color="blue", alpha=0.7)
    plt.xlabel("Objectness Score")
    plt.ylabel("Frequency")
    plt.title("Distribution of Objectness Scores")
    plt.grid(True)
    plt.yscale("log")
    if save_path:
        plt.savefig(os.path.join(save_path, save_file_name))
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
            color="lime",
        )

        ax.text(
            cx - w / 2 + 0.015,
            cy + h / 2 - 0.015,
            f"objectness: {objectness:1.2f}",
            ha="left",
            va="bottom",
            color="black",
            bbox={
                "facecolor": "white",
                "edgecolor": "lime",
                "boxstyle": "square,pad=.3",
            },
        )

    ax.set_xlim(0, 1)
    ax.set_ylim(1, 0)
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, f"{image_name}_bounding_boxes.png"))
    plt.close(fig)

def plot_correlation_objectness_bbox_size(objectnesses, source_boxes, image_name, image_shape, save_path=None):
    bbox_sizes = [(w * h) / (image_shape[0] * image_shape[1]) for _, _, w, h in source_boxes]
    xy = np.vstack([bbox_sizes, objectnesses])
    z = gaussian_kde(xy)(xy)
    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(bbox_sizes, objectnesses, c=z, cmap="viridis", alpha=0.7)
    plt.colorbar(scatter, label="Density")
    plt.xlabel("Normalized Bounding Box Size (Area)")
    plt.ylabel("Objectness Score")
    plt.title(f"Correlation between Objectness Scores and Bounding Box Sizes for {image_name}")
    plt.grid(True)
    plt.xscale("log")
    if save_path:
        plt.savefig(os.path.join(save_path, f"{image_name}_correlation_objectness_bbox_size_density.png"))
    else:
        plt.show()
    plt.close()

def plot_bbox_size_histogram(image_name, source_boxes, image_shape, save_path=None):
    bbox_sizes = [(w * h) / (image_shape[0] * image_shape[1]) for _, _, w, h in source_boxes]
    plt.figure(figsize=(10, 6))
    plt.hist(bbox_sizes, bins=50, color="blue", alpha=0.7)
    plt.xlabel("Normalized Bounding Box Size")
    plt.ylabel("Frequency")
    plt.title(f"Histogram of Normalized Bounding Box Sizes for {image_name}")
    plt.grid(True)
    plt.xscale("log")
    if save_path:
        plt.savefig(os.path.join(save_path, f"{image_name}_bbox_size_histogram.png"))
    else:
        plt.show()
    plt.close()
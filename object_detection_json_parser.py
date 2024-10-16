from file_utils import read_annotation_file

import json
from pathlib import Path
from matplotlib.patches import Patch
from collections import defaultdict
from tqdm import tqdm
import cv2
import os
from scipy.stats import pearsonr, gaussian_kde
import matplotlib.pyplot as plt
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import math


def draw_bounding_boxes_and_save(frame, frame_name, video_name, objs, output_base_path, threshold=0.5):
    save_frame = False
    for obj in objs:
        bbox = obj["box"]
        class_name = obj["class"]
        conf = obj["confidence"]
        if conf > threshold:
            save_frame = True
            xmin, ymin, xmax, ymax = map(int, bbox)
            if class_name == "gun":
                color = (0, 0, 255)
            elif class_name == "knife":
                color = (0, 255, 0)
            else:
                color = (255, 0, 0)
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 1)
            cv2.putText(frame, f"{class_name}-{conf}", (xmin, ymin), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
    if save_frame:
        save_folder = Path(output_base_path, video_name, str(threshold))
        os.makedirs(save_folder.as_posix(), exist_ok=True)
        cv2.imwrite((save_folder / frame_name).as_posix(), frame)
        # close the window
        cv2.destroyAllWindows()


def load_json_files(json_path: str) -> dict:
    data = {}
    # get all json files in the directory
    json_files = [f for f in Path(json_path).rglob("*.json")]
    for json_file in tqdm(json_files):
        data[Path(json_file).stem] = read_annotation_file(json_file)
    return data


# Example usage
def calculate_normalized_correlation(annotation_dir_path: Path, save_path=None):
    annotation_files = [f for f in os.listdir(annotation_dir_path) if f.endswith(".json")]
    sizes = []
    confidences = []

    for annotation_file in tqdm(annotation_files):
        annotation = read_annotation_file(annotation_dir_path / Path(annotation_file))
        for frame in annotation.values():
            image_shape = frame["imagesize"]
            for obj in frame["objects"]:
                bbox = obj["box"]
                conf = obj["confidence"]
                xmin, ymin, xmax, ymax = map(int, bbox)
                width = xmax - xmin
                height = ymax - ymin
                obj_size = (width * height) / (image_shape[0] * image_shape[1])
                sizes.append(obj_size)
                confidences.append(conf)

    # Calculate correlation
    correlation, _ = pearsonr(sizes, confidences)
    print(f"Correlation between normalized object size and confidence score: {correlation}")

    # Calculate the point density
    xy = np.vstack([sizes, confidences])
    z = gaussian_kde(xy)(xy)

    # Plot the data
    plt.scatter(sizes, confidences, c=z, cmap="viridis", alpha=0.5)
    plt.colorbar(label="Density")
    plt.xlabel("Normalized Object Size")
    plt.ylabel("Confidence Score")
    plt.title("Correlation between Normalized Object Size and Confidence Score")
    plot_name = "normalized_correlation.png"
    if save_path:
        plt.savefig(Path(save_path,plot_name).as_posix())
    else:
        plt.show()
    plt.close()


def count_detections_per_class(annotation_dir_path: Path):
    annotation_files = [f for f in os.listdir(annotation_dir_path) if f.endswith(".json")]
    class_counts = defaultdict(int)

    for annotation_file in tqdm(annotation_files):
        annotation = read_annotation_file(annotation_dir_path / Path(annotation_file))
        for frame in annotation.values():
            for obj in frame["objects"]:
                class_name = obj["class"]
                class_counts[class_name] += 1

    return dict(class_counts)


def plot_detections_per_class(class_counts, save_path=None):
    classes = list(class_counts.keys())
    counts = list(class_counts.values())

    plt.bar(classes, counts, color=["blue", "green"])
    plt.xlabel("Class")
    plt.ylabel("Number of Detections")
    plt.title("Number of Detections per Class (Rifle vs. Knife)")
    plot_name = "detections_per_class.png"
    if save_path:
        plt.savefig(Path(save_path,plot_name).as_posix())
    else:
        plt.show()
    plt.close()


def count_detections_per_video(annotation_dir_path: Path):
    annotation_files = [f for f in os.listdir(annotation_dir_path) if f.endswith(".json")]
    video_detections = {}

    for annotation_file in tqdm(annotation_files):
        annotation = read_annotation_file(annotation_dir_path / Path(annotation_file))
        video_name = Path(annotation_file).stem
        detection_count = sum(len(frame["objects"]) for frame in annotation.values())
        video_detections[video_name] = detection_count

    return video_detections


def plot_detections_histogram(video_detections, save_path=None):
    detection_counts = list(video_detections.values())

    plt.hist(detection_counts, bins=10, color="blue", alpha=0.7)
    plt.xlabel("Number of Detections")
    plt.ylabel("Number of Videos")
    plt.title("Histogram of Detections per Video")
    plot_name = "detections_per_video.png"
    if save_path:
        plt.savefig(Path(save_path,plot_name).as_posix())
    else:
        plt.show()
    plt.close()


def extract_confidence_scores(annotation_dir_path: Path):
    annotation_files = [f for f in os.listdir(annotation_dir_path) if f.endswith(".json")]
    confidence_scores = defaultdict(list)

    for annotation_file in tqdm(annotation_files):
        annotation = read_annotation_file(annotation_dir_path / Path(annotation_file))
        for frame in annotation.values():
            for obj in frame["objects"]:
                class_name = obj["class"]
                conf = obj["confidence"]
                confidence_scores[class_name].append(conf)

    return dict(confidence_scores)


def plot_confidence_histogram(confidence_scores, save_path=None):
    plt.figure(figsize=(10, 10))
    cmap = plt.get_cmap("tab20")  # Use a colormap with distinct colors

    for i, (class_name, scores) in enumerate(confidence_scores.items()):
        plt.hist(scores, bins=10, alpha=0.5, label=class_name, color=cmap(i))

    plt.xlabel("Confidence Score")
    plt.ylabel("Number of Detections")
    plt.title("Histogram of Confidence Scores by Class")
    plt.legend(loc="upper right")
    plot_name = "confidence_histogram.png"
    if save_path:
        plt.savefig(Path(save_path, plot_name).as_posix())
    else:
        plt.show()
    plt.close()


def extract_class_frames(annotation_file: Path, threshold=0.5, class_name="gun"):
    class_frames = []

    annotation = read_annotation_file(annotation_dir_path / Path(annotation_file))
    for frame_name, frame in annotation.items():
        for obj in frame["objects"]:
            if obj["class"] == class_name and obj["confidence"] > threshold and obj["confidence"] <= threshold + 0.1:
                # frame_name, frame_data, video_name
                class_frames.append((frame_name, frame, annotation_file.stem.split("_object_detection")[0]))
                break

    return class_frames


def draw_bounding_boxes(frame, objs, threshold=0.5, class_name="gun"):
    for obj in objs:
        if obj["class"] == class_name and obj["confidence"] > threshold and obj["confidence"] <= threshold + 0.1:
            bbox = obj["box"]
            xmin, ymin, xmax, ymax = map(int, bbox)
            color = (0, 0, 255)
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 2)
            cv2.putText(frame, class_name, (xmin, ymin - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            cv2.putText(frame, f"{obj['confidence']:.2f}", (xmin, ymin - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return frame


def create_collage_plot(frames, images_base_dir, class_name, threshold=0.2, output_path=None):
    all_frames = []
    for frame_name, frame_data, video_name in frames:
        frame = cv2.imread(str(os.path.join(images_base_dir.as_posix(), video_name, "sampled_images", frame_name)))
        if frame is not None:
            frame = draw_bounding_boxes(frame, frame_data["objects"], threshold=threshold, class_name=class_name)
            all_frames.append(frame)

    if all_frames:
        # Determine the number of rows and columns needed
        total_images = len(all_frames)
        num_cols = math.ceil(math.sqrt(total_images))
        num_rows = math.ceil(total_images / num_cols)

        # Get the dimensions of the images
        img_height, img_width, _ = all_frames[0].shape
        white_space_col = 2  # 2-column width of white space
        white_space_row = 2  # 2-row height of white space

        # Create a blank image with the appropriate size
        row_height = img_height + white_space_row
        row_width = (img_width + white_space_col) * num_cols - white_space_col
        long_image = np.ones((num_rows * row_height - white_space_row, row_width, 3), dtype=np.uint8) * 255

        for idx, frame in enumerate(all_frames):
            row_idx = idx // num_cols
            col_idx = idx % num_cols
            start_x = col_idx * (img_width + white_space_col)
            start_y = row_idx * row_height
            long_image[start_y : start_y + img_height, start_x : start_x + img_width] = frame

        cv2.imwrite(output_path, long_image)


def draw_bounding_boxes_for_class_and_confidence_interval(annotation_file, images_base_dir, output_save_base_path, class_name, threshold=0.2):
    frames = extract_class_frames(annotation_file, threshold, class_name)
    # print(f"images_base_dir: {images_base_dir}")

    output_path = Path(output_save_base_path, f"{class_name}_{Path(annotation_file).stem}.png")
    # print(f"output_path: {output_path}")
    create_collage_plot(frames, images_base_dir, class_name, threshold, output_path)


def draw_bounding_boxes_for_class_and_confidence_intervals(images_base_dir, annotation_dir_path, class_name, output_base_path, run_parallel=True):
    annotation_files = [f for f in annotation_dir_path.iterdir() if f.suffix == ".json"]
    thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    for threshold in thresholds:
        th_str = str(threshold).replace(".", "_")
        output_save_base_path = Path(output_base_path, f"{class_name}_1000/detection_threshold_{th_str}")
        os.makedirs(output_save_base_path, exist_ok=True)
        if run_parallel:
            with ThreadPoolExecutor() as executor:
                list(
                    tqdm(
                        executor.map(
                            lambda f: draw_bounding_boxes_for_class_and_confidence_interval(
                                f, images_base_dir, output_save_base_path, class_name, threshold
                            ),
                            annotation_files,
                        ),
                        total=len(annotation_files),
                    )
                )
        else:
            for annotation_file in tqdm(annotation_files):
                draw_bounding_boxes_for_class_and_confidence_interval(annotation_file, images_base_dir, output_save_base_path, class_name, threshold)


def extract_detection_locations(annotation_dir_path: Path):
    annotation_files = [f for f in annotation_dir_path.iterdir() if f.suffix == ".json"]
    all_coords = []

    for annotation_file in tqdm(annotation_files):
        annotation = read_annotation_file(annotation_file)
        for frame in annotation.values():
            frame_width, frame_height = frame["imagesize"]
            for obj in frame["objects"]:
                bbox = obj["box"]
                xmin, ymin, xmax, ymax = map(int, bbox)
                # Normalize coordinates
                norm_xmin = xmin / frame_width
                norm_ymin = ymin / frame_height
                norm_xmax = xmax / frame_width
                norm_ymax = ymax / frame_height
                all_coords.append((norm_xmin, norm_ymin, norm_xmax, norm_ymax))

    return all_coords

def generate_heatmap(coords, frame_shape, save_path):
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    heatmap = np.zeros(frame_shape[:2], dtype=np.float32)

    for (norm_xmin, norm_ymin, norm_xmax, norm_ymax) in coords:
        xmin = int(norm_xmin * frame_shape[1])
        ymin = int(norm_ymin * frame_shape[0])
        xmax = int(norm_xmax * frame_shape[1])
        ymax = int(norm_ymax * frame_shape[0])
        heatmap[ymin:ymax, xmin:xmax] += 1

    # Normalize the heatmap values
    # heatmap = cv2.normalize(heatmap, None, 0, 1, cv2.NORM_MINMAX)

    fig, ax = plt.subplots()
    im = ax.imshow(heatmap, cmap="viridis")
    ax.set_title("Heatmap of Detection Locations")

    # Create a divider for the existing axes instance
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)

    # Create the colorbar
    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label('Detection Density')

    plot_name = "detection_location_heatmap.png"
    if save_path:
        plt.savefig(Path(save_path, plot_name).as_posix())
    else:
        plt.show()
    plt.close()


def extract_bbox_area_and_confidence(annotation_dir_path: Path):
    annotation_files = [f for f in annotation_dir_path.iterdir() if f.suffix == ".json"]
    data = []

    for annotation_file in tqdm(annotation_files):
        annotation = read_annotation_file(annotation_file)
        for frame in annotation.values():
            frame_width, frame_height = frame["imagesize"]
            for obj in frame["objects"]:
                bbox = obj["box"]
                confidence = obj["confidence"]
                class_name = obj["class"]
                xmin, ymin, xmax, ymax = map(int, bbox)
                # Calculate normalized area
                area = (xmax - xmin) * (ymax - ymin) / (frame_width * frame_height)
                data.append((area, confidence, class_name))

    return data

def plot_scatter(data, save_path):
    areas, confidences, classes = zip(*data)
    unique_classes = list(set(classes))
    colors = plt.colormaps.get_cmap('tab10')

    fig, ax = plt.subplots(figsize=(15, 10))  # Increase the figure size
    for i, class_name in enumerate(unique_classes):
        class_areas = [area for area, cls in zip(areas, classes) if cls == class_name]
        class_confidences = [conf for conf, cls in zip(confidences, classes) if cls == class_name]
        ax.scatter(class_areas, class_confidences, label=class_name, color=colors(i), alpha=0.5, s=8)  # Add transparency

    ax.set_xlabel('Normalized Bounding Box Area')
    ax.set_ylabel('Confidence Score')
    ax.set_title('Scatter Plot of Normalized Bounding Box Area vs. Confidence Score')
    ax.legend()
    plt.tight_layout()

    plot_name = "area_confidence_scatter.png"
    if save_path:
        plt.savefig(Path(save_path, plot_name).as_posix())
    else:
        plt.show()
    plt.close()
def extract_width_height_confidence(annotation_dir_path: Path):
    annotation_files = [f for f in os.listdir(annotation_dir_path) if f.endswith('.json')]
    data = []

    for annotation_file in tqdm(annotation_files):
        annotation = read_annotation_file(annotation_dir_path / Path(annotation_file))
        for frame in annotation.values():
            frame_width, frame_height = frame['imagesize']
            for obj in frame['objects']:
                bbox = obj['box']
                confidence = obj['confidence']
                xmin, ymin, xmax, ymax = map(int, bbox)
                width = (xmax - xmin) / frame_width
                height = (ymax - ymin) / frame_height
                data.append((width, height, confidence))

    return data

def plot_heatmap(data, save_path=None):
    widths, heights, confidences = zip(*data)
    heatmap, xedges, yedges = np.histogram2d(widths, heights, bins=50, weights=confidences, density=True)
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]

    plt.figure(figsize=(10, 8))
    plt.imshow(heatmap.T, extent=extent, origin='lower', cmap='viridis', aspect='auto')
    plt.colorbar(label='Confidence')
    plt.xlabel('Normalized Width')
    plt.ylabel('Normalized Height')
    plt.title('Heatmap of Normalized Width vs. Height with Confidence')

    plot_name = 'width_height_confidence_heatmap.png'
    if save_path:
        plt.savefig(Path(save_path, plot_name).as_posix())
    else:
        plt.show()
    plt.close()


def categorize_size(area):
    # Small: less than 1% of the frame area
    if area < 0.01:
        return 'small'
    # Medium: between 1% and 10% of the frame area
    elif area < 0.1:
        return 'medium'
    # Large: more than 10% of the frame area
    else:
        return 'large'
def plot_combined_box_plots(data, save_path=None):
    size_categories = {'small': [], 'medium': [], 'large': []}
    class_names = set()

    for area, confidence, class_name in data:
        size_category = categorize_size(area)
        size_categories[size_category].append((confidence, class_name))
        class_names.add(class_name)

    class_names = sorted(class_names)
    size_labels = ['small', 'medium', 'large']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']  # Colors for small, medium, large

    fig, ax = plt.subplots(figsize=(18, 6))
    box_data = []
    positions = []
    labels = []
    pos = 1
    class_mid_positions = []

    for class_name in class_names:
        class_positions = []
        class_has_data = False
        for size_label in size_labels:
            class_confidences = [conf for conf, cls in size_categories[size_label] if cls == class_name]
            if class_confidences:
                box_data.append(class_confidences)
                positions.append(pos)
                class_positions.append(pos)
                class_has_data = True
            pos += 1
        if class_has_data:
            labels.append(class_name)
            class_mid_positions.append(sum(class_positions) / len(class_positions))
            ax.axvline(x=pos - 0.5, color='gray', linestyle='--', alpha=0.5)  # Add a vertical line between classes
        pos += 1  # Add space between different classes

    box = ax.boxplot(box_data, positions=positions, patch_artist=True, medianprops=dict(color="red"))
    for patch, color in zip(box['boxes'], colors * len(class_names)):
        patch.set_facecolor(color)

    # Set tick positions to the middle of each class group
    ax.set_xticks(class_mid_positions)
    ax.set_xticklabels(labels, rotation=45, ha='center')

    # Adjust the xlim to ensure all boxes are visible
    ax.set_xlim(0, pos)

    legend_patches = [Patch(color=color, label=size_label.capitalize()) for color, size_label in zip(colors, size_labels)]
    ax.legend(handles=legend_patches, loc='upper right')
    ax.set_title('Confidence Distribution by Size Category and Class')
    ax.set_xlabel('Class')
    ax.set_ylabel('Confidence Score')

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plot_name = 'combined_confidence_distribution_box_plots.png'
    if save_path:
        plt.savefig(Path(save_path, plot_name).as_posix())
    else:
        plt.show()
    plt.close()


def plot_box_plots(data, save_path=None):

    size_categories = {'small': [], 'medium': [], 'large': []}
    class_names = set()

    for area, confidence, class_name in data:
        size_category = categorize_size(area)
        size_categories[size_category].append((confidence, class_name))
        class_names.add(class_name)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)
    size_labels = ['small', 'medium', 'large']
    size_descriptions = [
        '< 1% frame area',
        '1%-10% frame area',
        '> 10% frame area'
    ]
    colors = plt.cm.tab20.colors  # Use a colormap with distinct colors

    for ax, size_label, size_desc in zip(axes, size_labels, size_descriptions):
        box_data = []
        labels = []
        for class_name in class_names:
            class_confidences = [conf for conf, cls in size_categories[size_label] if cls == class_name]
            box_data.append(class_confidences)
            labels.append(class_name)
        box = ax.boxplot(box_data, labels=labels, patch_artist=True)
        for patch, color in zip(box['boxes'], colors):
            patch.set_facecolor(color)
        ax.set_title(f'{size_label.capitalize()} Objects\n({size_desc})')
        ax.set_xticklabels(labels, rotation=45, ha='right')
        ax.set_xlabel('Class')
        ax.set_ylabel('Confidence Score')

    plt.suptitle('Confidence Distribution by Size Category and Class')
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plot_name = 'confidence_distribution_box_plots.png'
    if save_path:
        plt.savefig(Path(save_path, plot_name).as_posix())
    else:
        plt.show()
    plt.close()


def plot_regression_with_density(data, save_path=None):
    import seaborn as sns
    import pandas as pd
    from scipy.stats import gaussian_kde
    df = pd.DataFrame(data, columns=['area', 'confidence', 'class'])
    classes = df['class'].unique()

    for class_name in classes:
        class_data = df[df['class'] == class_name]
        x = class_data['area']
        y = class_data['confidence']

        # Calculate the point density
        xy = np.vstack([x, y])
        z = gaussian_kde(xy)(xy)

        plt.figure(figsize=(10, 6))
        scatter = plt.scatter(x, y, c=z, s=10, cmap='viridis', alpha=0.5)
        plt.colorbar(scatter, label='Density')
        sns.regplot(x='area', y='confidence', data=class_data, scatter=False, line_kws={'color': 'red'})
        plt.xlabel('Normalized Area')
        plt.ylabel('Confidence Score')
        plt.title(f'Regression Plot with Density for {class_name.capitalize()}')
        plot_name = f'regression_plot_density_{class_name}.png'
        if save_path:
            os.makedirs(save_path, exist_ok=True)
            plt.savefig(Path(save_path, plot_name).as_posix())
        else:
            plt.show()
        plt.close()

if __name__ == "__main__":
    # Set the base directories
    proj_base_dir = Path("C:/Users/avich/projects/owl_vit_object_detection_evaluation")
    data_base_dir = Path("C:/Users/avich/projects/Data/objects_tracking_dataset/Ben_data")

    images_base_dir = Path(data_base_dir,"videos_frame_samples")
    annotation_dir_path = Path(data_base_dir,"obj_detection_json")
    output_base_path = Path(proj_base_dir, "results")
    plot_graph_base_path = Path(proj_base_dir,output_base_path, "plots")
    class_name = "rifle"

    # Bounding box area vs. confidence regression plot with density
    print("Bounding box area vs. confidence regression plot")
    data = extract_bbox_area_and_confidence(annotation_dir_path)
    plot_regression_with_density(data, plot_graph_base_path)

    # Box plots for confidence distribution by size category and class
    print("Box plots for confidence distribution by size category and class")
    data = extract_bbox_area_and_confidence(annotation_dir_path)
    plot_box_plots(data, plot_graph_base_path)
    # Width, height, and confidence heatmap
    print("Width, height, and confidence heatmap")
    data = extract_width_height_confidence(annotation_dir_path)
    plot_heatmap(data, plot_graph_base_path)

    # Scatter plot of normalized bounding box area vs. confidence score
    print("Scatter plot of normalized bounding box area vs. confidence score")
    data = extract_bbox_area_and_confidence(annotation_dir_path)
    plot_scatter(data, plot_graph_base_path)

    # Heatmap of detection locations
    print("Heatmap of detection locations")
    frame_shape = (1080, 1920, 3)  # Example frame shape, adjust as needed
    coords = extract_detection_locations(annotation_dir_path)
    generate_heatmap(coords, frame_shape, plot_graph_base_path)

    # draw_bounding_boxes_for_class_and_confidence_intervals(images_base_dir, annotation_dir_path, class_name, output_base_path)

    # detections per class histogram
    print("Detections per class histogram")
    class_counts = count_detections_per_class(annotation_dir_path)
    plot_detections_per_class(class_counts, plot_graph_base_path)

    # confidence histogram
    print("Confidence histogram")
    confidence_scores = extract_confidence_scores(annotation_dir_path)
    plot_confidence_histogram(confidence_scores, plot_graph_base_path)

    # detections per video histogram
    print("Detections per video histogram")
    video_detections = count_detections_per_video(annotation_dir_path)
    plot_detections_histogram(video_detections, plot_graph_base_path)

    # class_counts = count_detections_per_class(annotation_dir_path)
    # plot_detections_per_class(class_counts)
    # calculate_normalized_correlation(annotation_dir_path)
    # # Load multiple JSON files
    # json_path = 'C:/Users/avich/Projects/Data/objects_tracking_dataset/Ben_data/json_object_detection/knife_1_object_detection.json'
    # plot_path = 'C:/Users/avich/Projects/owl_vit_object_detection_evaluation/results/videos_with_annotation'
    # frames_path = 'C:/Users/avich/Projects/Data/objects_tracking_dataset/Ben_data/knifes/knife_1/sampled_images'
    # data = read_annotation_file(json_path)
    # ths_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    # # go over all frames and draw bounding boxes
    # for frame_name, details in tqdm(data.items()):
    #     for th in ths_list:
    #
    #         if len(details['objects'])> 0:
    #             frame = cv2.imread(f"{frames_path}/{frame_name}")
    #             draw_bounding_boxes_and_save(frame,frame_name,'knife_1',details['objects'],plot_path, threshold=th)
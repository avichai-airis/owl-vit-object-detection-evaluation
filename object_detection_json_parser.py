from file_utils import read_annotation_file

import json
from pathlib import Path
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
        Path(save_path,plot_name).as_posix()
    else:
        plt.show()


def count_detections_per_class(annotation_dir_path: Path):
    annotation_files = [f for f in os.listdir(annotation_dir_path) if f.endswith(".json")]
    class_counts = {"gun": 0, "knife": 0}

    for annotation_file in tqdm(annotation_files):
        annotation = read_annotation_file(annotation_dir_path / Path(annotation_file))
        for frame in annotation.values():
            for obj in frame["objects"]:
                class_name = obj["class"]
                if class_name in class_counts:
                    class_counts[class_name] += 1

    return class_counts


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
        Path(save_path,plot_name).as_posix()
    else:
        plt.show()


def extract_confidence_scores(annotation_dir_path: Path):
    annotation_files = [f for f in os.listdir(annotation_dir_path) if f.endswith(".json")]
    confidence_scores = {"gun": [], "knife": []}

    for annotation_file in tqdm(annotation_files):
        annotation = read_annotation_file(annotation_dir_path / Path(annotation_file))
        for frame in annotation.values():
            for obj in frame["objects"]:
                class_name = obj["class"]
                conf = obj["confidence"]
                if class_name in confidence_scores:
                    confidence_scores[class_name].append(conf)

    return confidence_scores


def plot_confidence_histogram(confidence_scores, save_path=None):
    plt.figure(figsize=(10, 5))

    for class_name, scores in confidence_scores.items():
        plt.hist(scores, bins=10, alpha=0.5, label=class_name)

    plt.xlabel("Confidence Score")
    plt.ylabel("Number of Detections")
    plt.title("Histogram of Confidence Scores by Class")
    plt.legend(loc="upper right")
    plot_name = "confidence_histogram.png"
    if save_path:
        Path(save_path,plot_name).as_posix()
    else:
        plt.show()


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


if __name__ == "__main__":
    # Set the base directories
    proj_base_dir = Path("/home/ubuntu/projects/owl-vit-object-detection-evaluation")
    data_base_dir = Path("/home/ubuntu/Data/obj_det_eval_dataset")

    images_base_dir = Path(data_base_dir,"videos_frame_samples")
    annotation_dir_path = Path(data_base_dir,"obj_detection_json")
    output_base_path = Path(proj_base_dir, "results")
    plot_graph_base_path = Path(proj_base_dir,output_base_path, "plots")
    class_name = "rifle"

    draw_bounding_boxes_for_class_and_confidence_intervals(images_base_dir, annotation_dir_path, class_name, output_base_path)

    # detections per class histogram
    class_counts = count_detections_per_class(annotation_dir_path)
    plot_detections_per_class(class_counts)

    # confidence histogram
    confidence_scores = extract_confidence_scores(annotation_dir_path)
    plot_confidence_histogram(confidence_scores)

    # detections per video histogram
    video_detections = count_detections_per_video(annotation_dir_path)
    plot_detections_histogram(video_detections)

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

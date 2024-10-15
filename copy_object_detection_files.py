import os
import shutil


def copy_object_detection_json(src_dir, dst_dir):
    # Ensure the destination directory exists
    os.makedirs(dst_dir, exist_ok=True)

    # Walk through each folder in the source directory
    for root, dirs, files in os.walk(src_dir):
        # Check if 'object_detection.json' exists in the current folder
        if "object_detection.json" in files:
            # Get the name of the folder (last part of the path)
            folder_name = os.path.basename(root)

            # Construct the full path to the source file
            src_file = os.path.join(root, "object_detection.json")

            # Create the destination file name (use the folder name)
            dst_file = os.path.join(dst_dir, f"{folder_name}_object_detection.json")

            # Copy the file and rename it
            shutil.copy(src_file, dst_file)
            print(f"Copied: {src_file} -> {dst_file}")


# Example usage:

source_directory = "/home/ubuntu/Data/obj_det_eval_dataset/videos_frame_samples"
destination_directory = "/home/ubuntu/Data/obj_det_eval_dataset/object_detection_1000_gun_and_knife_json"
copy_object_detection_json(source_directory, destination_directory)

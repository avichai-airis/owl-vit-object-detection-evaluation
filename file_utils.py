import json





def read_annotation_file(annotation_file) -> dict:
    with open(annotation_file) as f:
        annotations = json.load(f)
    return annotations['obj_detection']
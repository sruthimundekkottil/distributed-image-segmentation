"""Small helpers for COCO-style JSON (the format the segmentation model returns).

A COCO file has three main lists:
  images       [{"id", "file_name", "width", "height"}, ...]
  categories   [{"id", "name"}, ...]
  annotations  [{"id", "image_id", "category_id", "bbox", "segmentation", "score", ...}, ...]
Annotations point at their image through "image_id".
"""

import copy


def is_coco(result: dict) -> bool:
    """True if a model result looks like COCO (the classifier's output does not)."""
    return isinstance(result, dict) and "images" in result and "annotations" in result


def filter_by_score(coco: dict, min_score: float) -> dict:
    """Return a copy of a COCO dict that only keeps annotations with score >= min_score."""
    filtered = copy.copy(coco)  # shallow copy is enough: we only replace one list
    filtered["annotations"] = [a for a in coco["annotations"] if a.get("score", 1.0) >= min_score]
    return filtered


def merge(items: list[tuple[str, str, dict]], min_score: float = 0.0) -> dict:
    """Combine several single-image COCO dicts into one file.

    ``items`` is a list of (job_id, original_file_name, coco_dict). Each input
    uses image id 1 and annotation ids starting at 1, so we renumber them to
    keep every id unique in the combined file.
    """
    combined = {"images": [], "annotations": [], "categories": []}
    next_annotation_id = 1
    for image_id, (job_id, file_name, coco) in enumerate(items, start=1):
        if not combined["categories"]:
            combined["categories"] = coco.get("categories", [])
            combined["info"] = coco.get("info", {})

        source_image = coco["images"][0]
        combined["images"].append(
            {
                "id": image_id,
                "file_name": file_name,
                "width": source_image["width"],
                "height": source_image["height"],
                "job_id": job_id,  # extra field so you can trace an image back to its job
            }
        )
        for annotation in filter_by_score(coco, min_score)["annotations"]:
            combined["annotations"].append(
                {**annotation, "id": next_annotation_id, "image_id": image_id}
            )
            next_annotation_id += 1
    return combined

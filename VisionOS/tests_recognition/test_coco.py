"""Test parse ground-truth COCO (recognition.coco) — thuần, không cần mạng."""

from recognition.coco import (
    EVAL_CLASSES,
    coco_bbox_to_xyxy,
    parse_coco_gt,
    select_images_with_category,
)

COCO = {
    "categories": [
        {"id": 1, "name": "person"},
        {"id": 3, "name": "car"},
        {"id": 44, "name": "bottle"},
    ],
    "images": [
        {"id": 10, "file_name": "a.jpg", "coco_url": "http://x/a.jpg"},
        {"id": 20, "file_name": "b.jpg", "coco_url": "http://x/b.jpg"},
        {"id": 30, "file_name": "c.jpg"},
    ],
    "annotations": [
        {"image_id": 10, "category_id": 1, "bbox": [10, 10, 20, 20], "iscrowd": 0},
        {"image_id": 10, "category_id": 1, "bbox": [50, 50, 10, 10], "iscrowd": 0},
        {"image_id": 10, "category_id": 3, "bbox": [0, 0, 5, 5], "iscrowd": 0},   # car
        {"image_id": 20, "category_id": 1, "bbox": [1, 1, 2, 2], "iscrowd": 1},   # crowd
        {"image_id": 30, "category_id": 44, "bbox": [3, 3, 4, 4], "iscrowd": 0},
    ],
}


def test_bbox_conversion():
    assert coco_bbox_to_xyxy([10, 10, 20, 20]) == (10, 10, 30, 30)


def test_parse_gt_filters_and_groups():
    gt = parse_coco_gt(COCO, ["person"])
    assert set(gt) == {10}
    assert gt[10] == [(10, 10, 30, 30), (50, 50, 60, 60)]


def test_parse_gt_excludes_crowd_by_default():
    assert 20 not in parse_coco_gt(COCO, ["person"])
    assert 20 in parse_coco_gt(COCO, ["person"], include_crowd=True)


def test_parse_gt_by_name():
    assert parse_coco_gt(COCO, ["bottle"]) == {30: [(3, 3, 7, 7)]}


def test_select_images_and_limit():
    assert [im["id"] for im in select_images_with_category(COCO, ["person"], 100)] == [10]
    assert select_images_with_category(COCO, ["person"], 0) == []


def test_eval_classes_cover_three_tasks():
    assert {"person", "car", "bottle"} <= set(EVAL_CLASSES)
    assert EVAL_CLASSES["bottle"]["prompt"] == "bottle"

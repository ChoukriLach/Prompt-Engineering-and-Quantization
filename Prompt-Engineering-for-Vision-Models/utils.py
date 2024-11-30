import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import numpy as np
from PIL import Image
import random
import torch

def resize_image(image, input_size):
    w, h = image.size
    scale = input_size / max(w, h)
    new_w = int(w * scale)
    new_h = int(h * scale)
    image = image.resize((new_w, new_h))
    return image

def show_points_on_image(raw_image, input_points, input_labels=None):
    plt.figure(figsize=(10,10))
    plt.imshow(raw_image)
    input_points = np.array(input_points)
    if input_labels is None:
      labels = np.ones_like(input_points[:, 0])
    else:
      labels = np.array(input_labels)
    show_points(input_points, labels, plt.gca())
    plt.axis('on')
    plt.show()

def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)

def format_results(result, filter=0):
    annotations = []
    n = len(result.masks.data)
    for i in range(n):
        annotation = {}
        mask = result.masks.data[i] == 1.0

        if torch.sum(mask) < filter:
            continue
        annotation["id"] = i
        annotation["segmentation"] = mask.cpu().numpy()
        annotation["bbox"] = result.boxes.data[i]
        annotation["score"] = result.boxes.conf[i]
        annotation["area"] = annotation["segmentation"].sum()
        annotations.append(annotation)
    return annotations

def point_prompt(masks, points, point_label):  
    h = masks[0]["segmentation"].shape[0]
    w = masks[0]["segmentation"].shape[1]
    
    onemask = np.zeros((h, w))
    masks = sorted(masks, key=lambda x: x['area'], reverse=True)
    for i, annotation in enumerate(masks):
        if type(annotation) == dict:
            mask = annotation['segmentation']
        else:
            mask = annotation
        for i, point in enumerate(points):
            if mask[point[1], point[0]] == 1 and point_label[i] == 1:
                onemask[mask] = 1
            if mask[point[1], point[0]] == 1 and point_label[i] == 0:
                onemask[mask] = 0
    onemask = onemask >= 1
    return onemask, 0

def show_masks_on_image(image, masks):
    image_with_mask = image.convert("RGBA")
    
    for mask in masks:
        
        height, width = mask.shape
        mask_array = np.zeros((height, width, 4), dtype=np.uint8)
        color = [random.randint(0, 255), random.randint(0, 255), random.randint(0, 255), 150]
        
        mask_array[mask, :] = color
        mask_image = Image.fromarray(mask_array)

        width, height = image_with_mask.size
        mask_image = mask_image.resize((width, height))
        
        image_with_mask = Image.alpha_composite(
            image_with_mask,
            mask_image)
    
    image_with_mask.show()

def box_prompt(masks, bbox):
    h = masks.shape[1]
    w = masks.shape[2]
    
    bbox[0] = round(bbox[0]) if round(bbox[0]) > 0 else 0
    bbox[1] = round(bbox[1]) if round(bbox[1]) > 0 else 0
    bbox[2] = round(bbox[2]) if round(bbox[2]) < w else w
    bbox[3] = round(bbox[3]) if round(bbox[3]) < h else h

    bbox_area = (bbox[3] - bbox[1]) * (bbox[2] - bbox[0])

    masks_area = torch.sum(masks[:, bbox[1] : bbox[3], bbox[0] : bbox[2]], dim=(1, 2))
    orig_masks_area = torch.sum(masks, dim=(1, 2))

    union = bbox_area + orig_masks_area - masks_area
    IoUs = masks_area / union
    max_iou_index = torch.argmax(IoUs)

    return masks[max_iou_index].cpu().numpy(), max_iou_index

def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))

def show_boxes_on_image(raw_image, boxes):
    plt.figure(figsize=(10,10))
    plt.imshow(raw_image)
    for box in boxes:
      show_box(box, plt.gca())
    plt.axis('on')
    plt.show()

def preprocess_outputs(output):
    input_scores = [x["score"] for x in output]
    input_labels = [x["label"] for x in output]
    input_boxes = []
    for i in range(len(output)):
        input_boxes.append([*output[i]["box"].values()])
    input_boxes = [input_boxes]
    return input_scores, input_labels, input_boxes

def show_boxes_and_labels_on_image(raw_image, boxes, labels, scores):
    plt.figure(figsize=(10, 10))
    plt.imshow(raw_image)
    for i, box in enumerate(boxes):
        show_box(box, plt.gca())
        plt.text(
            x=box[0],
            y=box[1] - 12,
            s=f"{labels[i]}: {scores[i]:,.4f}",
            c="beige",
            path_effects=[pe.withStroke(linewidth=4, foreground="darkgreen")],
        )
    plt.axis("on")
    plt.show()
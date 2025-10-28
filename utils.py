import torch
import numpy as np
import torch.nn.functional as F
from typing import Any, Dict
from pycocotools import mask as mask_utils

def polygons_to_rle(polygons, height, width):
    all_rles = []
    for polygon in polygons:
        flatten = [p for sublist in polygon for p in sublist]
        rles = mask_utils.frPyObjects([flatten], height, width)
        all_rles.extend(rles) 
    merged_rle = mask_utils.merge(all_rles)
    if len(all_rles) == 0:
        merged_rle["size"] = [height,width]
    return merged_rle

def area_from_rle(rle: Dict[str, Any]) -> int:
    return sum(rle["counts"][1::2])

def mask_to_rle_pytorch(tensor: torch.Tensor):
    """
    Encodes masks to an uncompressed RLE, in the format expected by
    pycoco tools.
    """
    # Put in fortran order and flatten h,w
    b, h, w = tensor.shape
    tensor = tensor.permute(0, 2, 1).flatten(1)

    # Compute change indices
    diff = tensor[:, 1:] ^ tensor[:, :-1]
    change_indices = diff.nonzero()

    # Encode run length
    out = []
    for i in range(b):
        cur_idxs = change_indices[change_indices[:, 0] == i, 1]
        cur_idxs = torch.cat(
            [torch.tensor([0], dtype=cur_idxs.dtype, device=cur_idxs.device), cur_idxs + 1,
             torch.tensor([h * w], dtype=cur_idxs.dtype, device=cur_idxs.device), ]
        )
        btw_idxs = cur_idxs[1:] - cur_idxs[:-1]
        counts = [] if tensor[i, 0] == 0 else [0]
        counts.extend(btw_idxs.detach().cpu().tolist())
        out.append({"size": [h, w], "counts": counts})

    return out

def mask_to_rle_numpy(mask: np.ndarray):
    """
    Encodes masks to an uncompressed RLE, in the format expected by
    pycoco tools.
    """
    h, w = mask.shape

    # Put in fortran order and flatten h,w
    mask = np.transpose(mask).flatten()

    # Compute change indices
    diff = mask[1:] ^ mask[:-1]
    change_indices = np.where(diff)[0]

    # Encode run length
    cur_idxs = np.concatenate(
        ([0], change_indices + 1, [h * w])
    )
    btw_idxs = cur_idxs[1:] - cur_idxs[:-1]
    counts = [] if mask[0] == 0 else [0]
    counts.extend(btw_idxs.tolist())

    return {"size": [h, w], "counts": counts}

def coco_encode_rle(uncompressed_rle):
    h, w = uncompressed_rle["size"]
    rle = mask_utils.frPyObjects(uncompressed_rle, h, w)
    rle["counts"] = rle["counts"].decode("utf-8")  # Necessary to serialize with json

    return rle

def compute_iou(mask1, mask2):
    intersection = np.logical_and(mask1, mask2)
    union = np.logical_or(mask1, mask2)
    iou = np.sum(intersection) / np.sum(union)
    return iou


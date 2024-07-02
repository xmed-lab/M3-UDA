import torch
import torchvision.transforms as transforms
import cv2
import numpy as np
import torch.nn.functional as F

def match_histogram(source_img, target_img):

    hist_source = torch.histc(source_img, bins=256, min=0, max=255)
    hist_target = torch.histc(target_img, bins=256, min=0, max=255)

    hist_source = hist_source / hist_source.sum()
    hist_target = hist_target / hist_target.sum()

    cumulative_source = torch.cumsum(hist_source, dim=0)
    cumulative_target = torch.cumsum(hist_target, dim=0)
    mapping_table = torch.zeros(256)

    for i in range(256):
        mapping_table[i] = torch.argmin(torch.abs(cumulative_source - cumulative_target[i]))
    
    matched_image = mapping_table[source_img.to(torch.int64)].to(torch.float32)

    return matched_image
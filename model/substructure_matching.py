from os.path import join
from numpy.core.fromnumeric import size
import torch
import torch.nn.functional as F
from tqdm import tqdm
from utils.config import opt
from model.sinkhorn_distance import SinkhornDistance



def substructure_matching_sinkhorn(targets_src, boxes_t, label_num):
    # Get the number of nodes, node categories and node bounding box
    n = len(targets_src.fields['labels'])
    if n != label_num:
        return torch.tensor(0,device=opt.device,dtype=float)
    label_s = targets_src.fields['labels']
    box_s = targets_src.box
    label_t = boxes_t.fields['labels']
    box_t = boxes_t.box
    # Handle the sorting of sources
    sorted_indices = torch.argsort(label_s)
    label_s = label_s[sorted_indices]
    box_s = box_s[sorted_indices]
    # Gets the node with the highest score target
    arr = []
    for i in range(label_t.min(),label_t.max()+1):
        l = torch.where(label_t == i)
        if l is not None and l[0].shape[0] != 0:
            arr.append(l[0][0].item())

    label_t = label_t[arr]
    box_t = box_t[arr]

    if label_t.shape[0] != label_num:
        return torch.tensor(0,device=opt.device,dtype=float)

    # Get the middle point
    midpoint_s = []
    midpoint_t = []
    for i in range(n):
        midpoint_s.append([(box_s[i][0]+box_s[i][2])/2, (box_s[i][1]+box_s[i][3])/2])
        midpoint_t.append([(box_t[i][0]+box_t[i][2])/2, (box_t[i][1]+box_t[i][3])/2])
    
    coordinates_s = torch.tensor(midpoint_s)
    coordinates_t = torch.tensor(midpoint_t)
    # obtain the angular adjacency matrix
    adjacency_matrix_s = torch.zeros(label_num, label_num).to(device=opt.device)
    adjacency_matrix_t = torch.zeros(label_num, label_num).to(device=opt.device)

    # Fill the adjacency matrix, storing the Angle size
    for i in range(label_num):
        for j in range(label_num):
            if i != j:
                angle_s = calculate_angle(coordinates_s[i], coordinates_s[j])
                angle_t = calculate_angle(coordinates_t[i], coordinates_t[j])
                adjacency_matrix_s[i][j] = angle_s
                adjacency_matrix_t[i][j] = angle_t
    
    
    sinkhornDistance = SinkhornDistance(0.001, 120)
    loss, _, _ = sinkhornDistance(adjacency_matrix_s, adjacency_matrix_t)
    
    return loss[0]


def substructure_matching_L2(targets_src, boxes_t, label_num):
    # Get the number of nodes, node categories and node bounding box
    n = len(targets_src.fields['labels'])
    if n != label_num:
        return torch.tensor(0,device=opt.device,dtype=float)
    label_s = targets_src.fields['labels']
    box_s = targets_src.box
    label_t = boxes_t.fields['labels']
    box_t = boxes_t.box
    # Handle the sorting of sources
    sorted_indices = torch.argsort(label_s)
    label_s = label_s[sorted_indices]
    box_s = box_s[sorted_indices]
    # Gets the node with the highest score target
    arr = []
    for i in range(label_t.min(),label_t.max()+1):
        l = torch.where(label_t == i)
        if l is not None and l[0].shape[0] != 0:
            arr.append(l[0][0].item())

    label_t = label_t[arr]
    box_t = box_t[arr]

    if label_t.shape[0] != label_num:
        return torch.tensor(0,device=opt.device,dtype=float)

    # Get the middle point
    midpoint_s = []
    midpoint_t = []
    for i in range(n):
        midpoint_s.append([(box_s[i][0]+box_s[i][2])/2, (box_s[i][1]+box_s[i][3])/2])
        midpoint_t.append([(box_t[i][0]+box_t[i][2])/2, (box_t[i][1]+box_t[i][3])/2])
    
    coordinates_s = torch.tensor(midpoint_s)
    coordinates_t = torch.tensor(midpoint_t)
    # obtain the angular adjacency matrix
    adjacency_matrix_s = torch.zeros(label_num, label_num).to(device=opt.device)
    adjacency_matrix_t = torch.zeros(label_num, label_num).to(device=opt.device)

    # Fill the adjacency matrix, storing the Angle size
    for i in range(label_num):
        for j in range(label_num):
            if i != j:
                angle_s = calculate_angle(coordinates_s[i], coordinates_s[j])
                angle_t = calculate_angle(coordinates_t[i], coordinates_t[j])
                adjacency_matrix_s[i][j] = angle_s
                adjacency_matrix_t[i][j] = angle_t
    
    
    diff = adjacency_matrix_s - adjacency_matrix_t
    squared_diff = diff.pow(2)
    loss = torch.sqrt(squared_diff.sum(dim=1))
    
    return sum(loss)

# Calculate angles (radians) between points
def calculate_angle(point1, point2):
    delta_x = point2[0] - point1[0]
    delta_y = point2[1] - point1[1]
    angle = torch.atan2(delta_y, delta_x)
    return angle
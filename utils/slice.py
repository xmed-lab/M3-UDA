

import torch


def slice_tensor(my_list):
    list1, list2, list3, list4 = [], [], [], []
    num_parts = 4 
    for tensor in my_list:
        
        sub_tensors = torch.split(tensor, 1, dim=0)

        for i in range(num_parts):
            if i == 0:
                list1.append(sub_tensors[i])
            elif i == 1:
                list2.append(sub_tensors[i])
            elif i == 2:
                list3.append(sub_tensors[i])
            else:
                list4.append(sub_tensors[i])
    return [list1, list2, list3, list4]
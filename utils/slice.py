import torch

def slice_tensor(my_list, num):
    # Create a list of empty lists to hold the sub-tensors
    lists = [[] for _ in range(num)]
    
    # Loop through each tensor in the input list
    for tensor in my_list:
        # Split the tensor into sub-tensors
        sub_tensors = torch.split(tensor, 1, dim=0)
        
        # Append each sub-tensor to the corresponding list
        for i in range(num):
            lists[i].append(sub_tensors[i])
            
    return lists
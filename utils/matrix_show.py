from PIL import Image
import numpy as np


def show_m(matrix,name):
    tensor_np = matrix.cpu().numpy()

    if name == 'M':
        tensor_np = tensor_np*100

    scaled_tensor = (tensor_np * 255).astype(np.uint8)

    image = Image.fromarray(scaled_tensor, mode='L')  # 'L' 表示灰度图像

    image.save(name+'.png')  # 保存为图像文件
import os
from PIL import Image

# 输入文件夹路径
input_folder = '/home/um202276680/ZCL-RGBD/SUNRGBD/Depth'  # 替换为实际文件夹路径
output_folder = '/home/um202276680/ZCL-RGBD/SUNRGBD/black'  # 替换为实际输出文件夹路径

# 确保输出文件夹存在
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# 遍历输入文件夹中的所有文件
for filename in os.listdir(input_folder):
    if filename.endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
        # 打开图片
        img_path = os.path.join(input_folder, filename)
        img = Image.open(img_path)
        
        # 创建与原图大小相同的全0图像
        zero_img = Image.new(img.mode, img.size, (0,) * len(img.getbands()))
        
        # 保存全0图像
        zero_img_path = os.path.join(output_folder, filename)
        zero_img.save(zero_img_path)

print("全0图像生成完成。")
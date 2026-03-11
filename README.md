# whdld2

## 需要修改的地方
config.yaml文件  
data.root 修改为数据集根目录  
data.images_dir 修改为原始图片目录相对于数据集根目录的位置  
data.masks_dir 修改为标签图片目录相对于数据集根目录的位置

## 
进入到项目根目录，执行以下命令  
**分割数据集**  
python3 utils/split.py  

**模型训练**  
python3 train.py  

**评估模型**  
python3 eval.py

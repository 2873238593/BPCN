# BPCN
The code is for the paper "BPCN:Bilateral progressive compensation network for lung infection image segmentation"

# model
链接: https://pan.baidu.com/s/1YynHgZQ8QqfWZRqqeW1H5A  密码: whw6
命名为数据集数量
1698为1600+98
big为大数据集
98为50+48

# Dataset
数据集使用前要修改config以及data路径
链接: https://pan.baidu.com/s/1OkpNfntlTcyhA1gknhId5Q  密码: lnvt


# Train
python net_run.py
修改stage  == 'train' 训练模型

# Test
修改 stage  == 'test' 测试模型,修改agent_seg.py中checkpoint_name为测试的pt文件对训练结果进行测试


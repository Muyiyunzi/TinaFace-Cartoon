TODO: English version not available yet.

## 致谢

此内容是基于一个高度封装的[vedadet](https://github.com/Media-Smart/vedadet)框架和在其上实现的一篇SOTA文章[TinaFace](https://github.com/Media-Smart/vedadet/tree/main/configs/trainval/tinaface)实现的。

没有他们的辛勤努力就不会有此篇内容。

## 介绍

基于以上内容，我们完成了一个custom dataset的TinaFace实现。以下内容将以北航的机器学习课程大作业为例，详细讲述如何将一个custom dataset植入该模块。该任务是一个动漫Face检测的任务，我们最终达到了0.61的AP@0.7成绩（因为显卡较差，我们只训练了30个epoch，并且没有调参，模型本身应该有较大的潜力）。

动漫Face的数据集可在[北航云盘](https://bhpan.buaa.edu.cn:443/link/F8282A2D99FF1884BDC1B36B31521540)获取，但竞赛网站需要内网才能访问提交，所以在此略去，有兴趣者可以自行划分验证集评估模型指标。

## 特点

TinaFace相对于其他如YOLO等模型的特点就是训练速度较慢，但效果更好。本人的显卡是Quadro P5000，训练速度大概是100个batch（400张图像）2分钟。跑完整个epoch（42500张图像）大约需要3个多小时，速度还是比较慢的。

## 从零开始的安装过程

TODO：Ubuntu系统安装、NVIDIA显卡驱动、CUDA驱动安装

### 我的环境

- OS: Ubuntu 18.04.5 LTS Bionic Beaver
- CUDA: 10.2
- PyTorch 1.7.1
- Python 3.8.5

### 搭建基本环境

a. 将git克隆到本地

```shell
git clone https://github.com/Muyiyunzi/TinaFace-Cartoon.git
```

b. 创建一个便于管理的conda环境（需要预先安装anaconda） , *e.g.*,

```shell
conda create -n vedadet python=3.8.5 -y
conda activate vedadet
```

可能的debug方案：sudo chown -R username anaconda3（cd到anaconda安装命令下）

c. 通过[官方建议](https://pytorch.org/)命令安装pytorch，比如：

```shell
conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch
```

d. 将路径挂载到vedadet文件夹根目录
```shell
cd TinaFace-Cartoon
vedadet_root=${PWD}
```

e. 安装vedadet相关模块
```shell
pip install -r requirements/build.txt
pip install -v -e .
```

（注意最后的“.”）

f. 将cartoonFace（一个custom数据集）拷贝到data/下

从[北航云盘](https://bhpan.buaa.edu.cn:443/link/F8282A2D99FF1884BDC1B36B31521540)下载数据集

g. 将各个图像的annotation（ground truth）拼接成一个txt文件，并利用xml tools转换为xml文件便于训练
```shell
bash annotrans.sh
```

或打开annotrans.sh文件手动输入其中命令。

可能的debug方案: sudo chown 777 ./xxx.sh

h. 生成训练文件名
```shell
bash create_train_list.sh
```

### 训练（单卡训练）
```shell
bash train.sh
```
或打开train.sh文件手动输入其中命令。

### 推断（单卡推断）

将需要推断的权重文件复制到根目录下
```shell
cp workdir/cartoonface/epoch_1_weights.pth ./weights.pth
```

带有可视化信息的推断：
```shell
bash infer_visual.sh
```

不带有可视化信息的推断
```shell
bash infer_novisual.sh
```

单张图像的推断
```shell
bash infer_single.sh
```


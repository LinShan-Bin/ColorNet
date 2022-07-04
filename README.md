# ColorNet: Matching Clothes with Text

清华大学2022年《模式识别》课程大作业

最终结果：Acc = 97.76% (3rd), EM = 93.72% (2nd)

## 1 文件结构

本项目的文件结构如下：

```
|-ColorNet
|   |-dataset
|   |   |-medium
|   |   |   |-train
|   |   |   |   |-...
|   |   |   |-test
|   |   |   |   |-...
|   |   |   |-test_all.json
|   |   |   |-train_all.json
|   |-pretrained_model
|   |   |-ConvX_tiny
|   |   |   |-checkpoint.pth
|   |   |   |-model_5.pth
|   |   |   |-logs
|   |   |   |   |-...
|   |   |-...
|   |-label_processor
|   |   |-...
|   |-model
|   |   |-...
|   |-utils
|   |   |-__init__.py
|   |   |-clothes_dataset.py
|   |   |-trainer.py
|   |-.gitignore
|   |-README.md
|   |-requirements.txt
|   |-generate_label_munkres.ipynb
|   |-generate_label.ipynb
|   |-generate_mask.py
|   |-generate_test_json.py
|   |-train_model.py
```

## 2 各文件作用

`label_processor`文件夹下的文件用于处理标签；`model`文件夹为一些尝试过的模型，最终没有使用；
`utils`文件夹下包含打包好的 Trainer 类以及 Dataset 类；`generate_label_munkres.ipynb`
文件用于生成最终的标签，`generate_label.ipynb`是不含匈牙利算法的简单实现；`generate_mask.py`
使用实例分割模型生成 mask，没有使用；`generate_test_json.py`用于生成保存了所有图片对应 logits
的 json 文件；`train_model.py`用于训练模型。

## 3 使用方法

**Step 1:** 安装依赖

``` bash
pip install -r requirements.txt
```

主要是 PyTorch、Torchvision 版本需支持 ConvNeXt 模型；munkres 包中包含匈牙利算法的改进版本。

**Step 2:** 训练模型

``` bash
python train_model.py
```

默认训练的模型是 ConvNeXt tiny。可改 main 函数中被注释的部分来训练其他模型。

**Step 3:** 推理

``` bash
python generate_test_json.py
```

然后运行 `generate_label_munkres.ipynb` 文件，生成最终的标签。

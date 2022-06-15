## 数据说明

官方提供了245days数据，原本是大致8：1：1划分，但是目前我取消了test集合的划分（因为在训练过程中的test没有意义，还是取得val score最好的结果模型），所以改成了9：1的train和val，test测试运行predict.py会使用单独的另外一组数据predict_data/下的数据（感觉从目前几次结果来看，这个test集合有一定的参考意义）

## KNN构图实验

| K值   | 本地测试score | 云端结果      |
| ---- | --------- | --------- |
| 5    | 47        | 41.60     |
| 6    | 44.094    | 40.881    |
| 7    | 46.68     |           |
| 8    | 51.777    |           |
| 9    | 49.483    |           |
| 10   | 46        | 41.34     |
| 11   | 47.44     | 没测        |
| 12   | 47.27     | -41.29063 |
| 13   | 48.70     |           |
| 14   | 44.828    | 41.962    |
| 15   | 44.1601   | 40.94     |
| 16   | 47.6659   | 41.479    |
| 17   | 51.091    |           |
| 18   | 46.302    | 41.15     |
| 19   | 47.524    |           |
| 20   | 46        | 没测        |
|      |           |           |

## 注意力头参数

构图采用时间序列相关性，KNN取15

| heads | 本地score | 云端score |
| ----- | ------- | ------- |
| 2     | 45.6439 |         |
| 4     | 45.2643 |         |
| 8     | 44.1601 | 40.94   |
|       |         |         |

## Transformer enc-dec层数

构图采用时间序列相关性，KNN取15，8head注意力

| 层数   | 本地score | 云端score |
| ---- | ------- | ------- |
| 1    | 49.051  |         |
| 2    | 44.1601 | 40.94   |
| 3    | 45.531  |         |
| 4    |         |         |
| 5    |         |         |
|      |         |         |



## 空间构图

### 1. 距离构图

根据官方给的location文件构图，计算每两个电站之间的距离，再取knn-15进行构图。

| 本地Score | 云端Score |
| ------- | ------- |
| 45.607  |         |
|         |         |
|         |         |







# KDDCup 22 Wind Power Forecasting with Spatial-Temporal Graph Transformer

## Introduction
Wind Power Forecasting (WPF) aims to accurately estimate the wind power supply of a wind farm at different time scales. 
Wind power is one of the most installed renewable energy resources in the world, and the accuracy of wind power forecasting method directly affects dispatching and operation safety of the power grid.
WPF has been widely recognized as one of the most critical issues in wind power integration and operation. 


## Data Description
Please refer to KDD Cup 2022 --> Wind Power Forecast --> Task Definition 
(https://aistudio.baidu.com/aistudio/competition/detail/152/0/introduction)

Download data and place it into `./data`


## Model Training and Testing with the demo script

Minimum usage:
```
    CUDA_VISIBLE_DEVICES=0 python main.py  --conf config.yaml

    # make prediction with toy data
    # put the prediction data into  ./predict_data/test_x and ./predict_data/test_y
    CUDA_VISIBLE_DEVICES=0 python predict.py  --conf config.yaml
```
The trained model will be saved in `output_path` directory. Our model cost about 5 minutes for each epoch tested in Tesla V100 GPU.

## Requirements

```
pgl==2.2.3post0
paddlepaddle-gpu>=2.2.2
```


​    
## Model Architecture

We simply design a model as descripted bellows.

<img src="./model_archi.png" alt="The Model Architecture of WPF" width="800">

## Performance

|        | Dev Score | Max-dev Test Score |
| ------ | --------- | ------------------ |
| Report | -         | 47.7               |
| Ours   | 38.93     | 46.83              |

## Prediction Visualization

During Training we visualize the prediction in devided validation and test set. See `val_vis.png` and `test_vis.png`

## Suggest Reading Materials

1. Autoformer: Decomposition Transformers with Auto-Correlation for Long-Term Series Forecasting. [[link]](https://arxiv.org/abs/2106.13008)
2. Wind Farm Power prediction with Graph Neural Network. [[link]](https://aifrenz.github.io/present_file/wind_farm_presentation.pdf)

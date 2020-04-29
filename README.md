# SFC<sup>2</sup>Net
This repository implements SFC<sup>2</sup>Net proposed in the work:

**High-Throughput Rice Density Estimation from Transplantation to Tillering Stages Using Deep Networks**

Liang Liu, [Hao Lu](https://sites.google.com/site/poppinace/), Yanan Li, and Zhiguo Cao.

Plant Phenomics, in submission

### Example
<p align="left">
  <img src="example/T0004_SD_20120515110242_01.png" width="425" title="Example1"/>
</p>

<p align="left">
  <img src="example/T0004_SD_20130726110633_02.png" width="425" title="Example1"/>
</p>

<p align="left">
  <img src="example/T0004_SD_20130728110630_01.png" width="425" title="Example1"/>
</p>

## Installation
The code has been tested on Python 3.7.3 and PyTorch 1.3.1. Please follow the
official instructions to configure your environment. See other required packages
in `requirements.txt`.


## RPC dataset
* Download the Rice Plant Counting (RPC) test dataset from: [BaiduYun (597 
Mb)](https://pan.baidu.com/s/12IDidkL267dpNSvNrcFRUQ) (code: cirv ) or [OneDrive (597 
Mb)](https://1drv.ms/u/s!AkNf_IPSDakh5zGoa6svOTC_Nmwr?e=nLkAlR)
* Unzip the dataset and move it into the `./data` folder, the path structure should look like this:
````
$./data/rice_datasets-test
├──── images
├──── label_mat
├──── test.txt
````

## Inference
**Pre-trained Model on RPC dataset**
* Download the model from: [BaiduYun (48.8 
Mb)](https://pan.baidu.com/s/1pWowlSpKdhg6l_9qET2yUw) (code: 9g8e) or [OneDrive (48.8 
Mb)](https://1drv.ms/u/s!AkNf_IPSDakh5zdqa5c8Co5QzB9y?e=SAiyly)
* Move the model into the folder which the patch structure is:
````
$./snapshots/rice/sfc2net
├──── model_best.pth.tar
````

**Evaluation**
```python
python hltest.py
```

## Benchmark Results

### Counting Results on PRC dataset
| Method              | Venue, Year           | Pretrained    | MAE    | MSE    | rMAE  | R<sup>2</sup> |
| :--:                | :--:                  | :--:          | :--:   | :--:   | :--:  | :--:          |
| MCNN                | CVPR   2016           | No            | 92.11  | 121.52 | 15.33 | 0.89          |
| TasselNetV2         | Plant Methods   2019  | No            | 59.39  | 95.80  | 7.86  | 0.91          |
| CSRNet              | CVPR   2018           | VGG16         | 49.22  | 74.58  | 7.47  | 0.91          |
| BCNet               | TCSVT  2019           | VGG16         | 31.28  | 49.82  | 4.76  | 0.96          |
| SFC<sup>2</sup>Net  | This Paper            | MixNet-L      | 25.60  | 37.94  | 4.12  | 0.98          |


### Comparison of Different Backbones
| Backbone            | MAE    | MSE    | rMAE | R<sup>2</sup> |   Parameters  | Top-1 Acc.|
| :--:                | :--:   | :--:   | :--: |      :--:     |    :--:       | :--: |
| ResNet18            | 31.82  | 66.80  | 4.66 |      0.93     |    12.6M      | 69.8 |
| ResNet34            | 34.42  | 61.58  | 4.95 |      0.94     |    22.7M      | 73.3 |
| ResNet50            | 30.94  | 67.52  | 4.45 |      0.92     |    44.5M      | 76.2 |
| ResNet101           | 35.53  | 56.26  | 4.99 |      0.95     |    63.5M      | 77.4 |
| ResNet152           | 32.20  | 67.77  | 4.71 |      0.93     |    79.2M      | 78.3 |
| EfficientNet-B0     | 36.65  | 70.74  | 5.30 |      0.92     |    5.8M       | 77.3 |
| EfficientNet-B1     | 27.51  | 42.80  | 4.14 |      0.97     |    8.3M       | 79.2 |
| EfficientNet-B2     | 30.54  | 53.65  | 4.48 |      0.95     |    9.7M       | 80.3 |
| EfficientNet-B3     | 30.76  | 54.52  | 4.44 |      0.95     |    13.0M      | 81.7 |
| EfficientNet-B4     | 28.06  | 52.17  | 4.24 |      0.95     |    20.3M      | 83.0 |
| EfficientNet-B5     | 27.36  | 41.91  | 4.16 |      0.97     |    31.6M      | 83.7 |
| EfficientNet-B6     | 29.96  | 50.03  | 4.42 |      0.96     |    44.6M      | 84.2 |
| EfficientNet-B7     | 27.15  | 40.79  | 3.96 |      0.97     |    68.3M      | 84.4 |
| VGG16               | 30.67  | 57.53  | 4.51 |      0.95     |    15.7M      | 71.6 |
| MixNet-L            | 25.60  | 37.94  | 4.12 |      0.98     |    6.3M       | 78.9 |



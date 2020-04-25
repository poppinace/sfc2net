# sfc2net
by Liang Liu， [Hao Lu](https://sites.google.com/site/poppinace/)

## Installation
The code has been tested on Python 3.7.4 and PyTorch 1.2.0. Please follow the
official instructions to configure your environment. See other required packages
in `requirements.txt` (pending).

## Test SFC2Net Model
**Rice Plant Counting**
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

**Pre-trained Model in RPC dataset**
* Download the model from: [BaiduYun (48.8 
Mb)](https://pan.baidu.com/s/1pWowlSpKdhg6l_9qET2yUw) (code: 9g8e) or [OneDrive (48.8 
Mb)](https://1drv.ms/u/s!AkNf_IPSDakh5zdqa5c8Co5QzB9y?e=SAiyly)
* Move the model into the folder which the patch structure is:
````
$./snapshots/rice/sfc2net
├──── model_best.pth.tar
````



### Rice Plant Counting
| Method              | Venue, Year           | Pretrained    | MAE    | MSE    | rMAE  | R<sup>2</sup> |
| :--:                | :--:                  | :--:          | :--:   | :--:   | :--:  | :--:          |
| MCNN                | CVPR   2015           | No            | 92.11  | 121.52 | 15.33 | 0.89          |
| TasselNetV2         | Plant Methods   2019  | No            | 59.39  | 95.80  | 7.86  | 0.91          |
| CSRNet              | CVPR   2018           | Yes           | 49.22  | 74.58  | 7.47  | 0.91          |
| BCNet               | TCSVT  2019           | Yes           | 31.28  | 49.82  | 4.76  | 0.96          |
| SFC<sup>2</sup>Net  | This Paper            | Yes           | 25.60  | 37.94  | 4.12  | 0.98          |


### Comparison of Different Backbones
| Backbone            | MAE    | MSE    | rMAE | R<sup>2</sup> |   Parameters  | Top-1|
| :--:                | :--:   | :--:   | :--: |      :--:     |    :--:       | :--: |
| ResNet18            | 31.82  | 66.80  | 4.66 |      0.93     |    12.6M      | 69.8 |
| ResNet34            | 34.42  | 61.58  | 4.95 |      0.94     |    22.7M      | 73.3 |
| ResNet50            | 30.94  | 67.52  | 4.45 |      0.92     |    44.5M      | 76.2 |
| ResNet101           | 35.53  | 56.26  | 4.99 |      0.95     |    63.5M      | 77.4 |
| ResNet152           | 32.20  | 67.77  | 4.71 |      0.93     |    79.2M      | 78.3 |
| EfficientNet0       | 36.65  | 70.74  | 5.30 |      0.92     |    5.8M       | 77.3 |
| EfficientNet1       | 27.51  | 42.80  | 4.14 |      0.97     |    8.3M       | 79.2 |
| EfficientNet2       | 30.54  | 53.65  | 4.48 |      0.95     |    9.7M       | 80.3 |
| EfficientNet3       | 30.76  | 54.52  | 4.44 |      0.95     |    13.0M      | 80.3 |
| EfficientNet4       | 28.06  | 52.17  | 4.24 |      0.95     |    20.3M      | 80.3 |
| EfficientNet5       | 27.36  | 41.91  | 4.16 |      0.97     |    31.6M      | 80.3 |
| EfficientNet6       | 29.96  | 50.03  | 4.42 |      0.96     |    44.6M      | 80.3 |
| EfficientNet7       | 27.15  | 40.79  | 3.96 |      0.97     |    68.3M      | 80.3 |
| VGG16               | 30.67  | 57.53  | 4.51 |      0.95     |    15.7M      | 71.6 |
| SFC<sup>2</sup>Net  | 25.60  | 37.94  | 4.12 |      0.98     |    6.3M       | 78.9 |



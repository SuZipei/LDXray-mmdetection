## Introduction

Our project extends the [MMDetection](https://github.com/open-mmlab/mmdetection) framework to specialize in dual-view X-ray imaging. Unlike single-view datasets, dual-view imaging captures objects from both vertical and side perspectives, enhancing detection accuracy by reducing occlusions and providing more comprehensive views of the objects.

For more details, you can view our LDXray dataset introduction [here](https://suzipei.github.io/LDXray/).


## How To Run

First, install our project following these steps:

```bash
git clone https://github.com/SuZipei/LDXray-mmdetection.git
cd LDXray-mmdetection-main
pip install -v -e .
```

For more detailed installation instructions, please refer to the [installation guide](https://mmdetection.readthedocs.io/en/latest/get_started.html).

Second, [download](https://www.kaggle.com/datasets/yuzheguocs/LDXray) our LDXray dataset and place it in the `LDXray/dual-view` directory.

We provide a function `Load2ImageFromFiles` in `mmdet/datasets/transforms/loading.py` to load dual-view images from the LDXray dataset. Additionally, we have rewritten `ImgDataPreprocessor` in `mmdet/models/data_preprocessors/data_preprocessor.py` to normalize dual-view data.

Third, write a configuration file for your model. Examples of configuration files used in our research can be found in `LDXray/config`.

Finally, train or test your model. Please refer to the [train and test guide](https://mmdetection.readthedocs.io/en/latest/user_guides/index.html#train-test) for detailed instructions.


## Citation

Please note that our dataset is built upon the MMDetection framework, please cite the MMDetection framework as follows:
```plaintext
@article{mmdetection,
title = {{MMDetection}: Open MMLab Detection Toolbox and Benchmark},
author = {Chen, Kai and Wang, Jiaqi and Pang, Jiangmiao and Cao, Yuhang and
Xiong, Yu and Li, Xiaoxiao and Sun, Shuyang and Feng, Wansen and
Liu, Ziwei and Xu, Jiarui and Zhang, Zheng and Cheng, Dazhi and
Zhu, Chenchen and Cheng, Tianheng and Zhao, Qijie and Li, Buyu and
Lu, Xin and Zhu, Rui and Wu, Yue and Dai, Jifeng and Wang, Jingdong
and Shi, Jianping and Ouyang, Wanli and Loy, Chen Change and Lin, Dahua},
journal = {arXiv preprint arXiv:1906.07155},
year = {2019}
}
```


## License

- For academic and non-commercial use only
- Apache License 2.0

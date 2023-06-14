Toward Real-World Light Field Super-Resolution
====
Zeyu Xiao, Ruisheng Gao, Yutong Liu, Yueyi Zhang, and Zhiwei Xiong. [Toward Real-World Light Field Super-Resolution](https://openaccess.thecvf.com/content/CVPR2023W/LFNAT/html/Xiao_Toward_Real-World_Light_Field_Super-Resolution_CVPRW_2023_paper.html). In CVPRW 2023. <br/>

## Dependencies
- This repository is based on [SAV_conv](https://github.com/Joechann0831/SAV_conv) 

## The LytroZoom dataset, pretrained models, and reconstruction results
<!-- - link1：[USTCDisk](https://rec.ustc.edu.cn/share/a59c25a0-0a7c-11ee-8f1b-813c91a6ec14)   password：3mnz -->
- link：[BaiduDisk](https://pan.baidu.com/s/1csY_ndQpaPet_CmwSaKtqw)   password：lfsr 


## Train the model

```
python OProjNet1_train_x2_LZ_P.py
python OProjNet1_train_x2_LZ_O.py
```

## Test the model
```
python OProjNet1_inference_LytroZoom_x2_P.py
python OProjNet1_inference_LytroZoom_x2_O.py
```


## Citation
```
@InProceedings{Xiao_2023_toward,
    author    = {Xiao, Zeyu and Gao, Ruisheng and Liu, Yutong and Zhang, Yueyi and Xiong, Zhiwei},
    title     = {Toward Real-World Light Field Super-Resolution},
    booktitle = {CVPRW},
    year      = {2023},
}

@InProceedings{Xiao_2023_cutmib,
    author    = {Xiao, Zeyu and Liu, Yutong and Gao, Ruisheng and Xiong, Zhiwei},
    title     = {CutMIB: Boosting Light Field Super-Resolution via Multi-View Image Blending},
    booktitle = {CVPR},
    year      = {2023},
}
```

## Contact
Any question regarding this work can be addressed to zeyuxiao1997@163.com or zeyuxiao@mail.ustc.edu.cn.
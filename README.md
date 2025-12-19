# MyStyle++: A Controllable Personalized Generative Prior

![teaser](docs/teaser.png)

### [Project Page](https://libingzeng.github.io/projects/mystyle++/mystyle++.html) | [Paper](https://arxiv.org/pdf/2306.04865.pdf)

[MyStyle++: A Controllable Personalized Generative Prior](https://libingzeng.github.io/projects/mystyle++/mystyle++.htm)  
 [Libing Zeng](https://libingzeng.github.io/)<sup>1</sup>, [Lele Chen](https://lelechen63.github.io/)<sup>2</sup>, [Yi Xu](https://www.linkedin.com/in/yi-xu-42654823/)<sup>3</sup>, [Nima Khademi Kalantari](https://www.cs.tau.ac.il/~dcor/)<sup>1</sup> 

<sup>1</sup> Texas A&M University, <sup>2</sup> Sony AI, <sup>3</sup> OPPO US Research Center

## Setup

Code was tested with Python 3.9.5, Pytorch 1.9.0 and CUDA 11.4.

We provide a yml file to easy setup of a conda environment. 


```
conda env create -f mystyle_pp_env.yml
```

Auxilary pre-trained models are required for some workflows and are listed below.

| Name                                                                                                      | Data Preprocessing | Training           | Description                                                                                    |
| --------------------------------------------------------------------------------------------------------- | ------------------ | ------------------ |:---------------------------------------------------------------------------------------------- |
| [arcface](https://drive.google.com/file/d/1KW7bjndL3QG3sxBbZxreGHigcCCpsDgn/view?usp=sharing)             | :heavy_check_mark: |                    | Face-recognition network taken from [TreB1eN](https://github.com/TreB1eN/InsightFace_Pytorch). |
| [dlib landmarks model](http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2)                   | :heavy_check_mark: |                    | Used for face alignment.                                                                       |
| [FFHQ StyleGAN](https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/ffhq.pkl)               |                    | :heavy_check_mark: | StyleGANv2 model trained on FFHQ.                                                              |
| [W Inversion Encoder](https://drive.google.com/file/d/1M-hsL3W_cJKs77xM1mwq2e9-J0_m7rHP/view?usp=sharing) |                    | :heavy_check_mark: | Taken from [HyperStyle](https://github.com/yuval-alaluf/hyperstyle).                           |


## Data Preprocess
We provide detailed scripts for data preprocessing in the folder named data_preprocess.
### General Process
```
0_align_face.py # align face images
1_image_quality.py # remove low quality images
2_duplicates.py # remove duplicated images
```

### Attribute-Related Process
```
4_age_prediction.py # obtain age attribute if needed
5_face_recognition_aws.py # obtain various attributes via Recognition API of AWS
6_rename_celebrities1.py # Rename images to align the implementation of reconstruct/tune_net_anchor.py
```


## Training a Controllable Personalized Generator
For the illustration of traning and testing, we use celebrities1/barack_obama as an example.
To adapt a pretrained domain-generator to a personalized-generator, run the training command below. 
```
./shell/train_anchor_celebrities1_yawpitchexp_bo.sh
```

## Test time - Applications

### Semantic Editing
To achieve semantic editing, run the following command:
```
./shell/test_anchor_celebrities1_bo_yawpitchexp_images_free.sh
```

### Image Synthesis
To generate personalized images, run the following command:
```
./shell/application_syn_random_anchor_celebrities1_bo.sh
```


### Image Enhancement

**For super-resolution**, run the following command:
```
./shell/test_1_anchor_project_app_sr_celebrities.sh
```

**For inpainting**, run the following command:
```
./shell/test_1_anchor_project_app_imp_celebrities.sh
```



## Citation

```
@article{Zeng_2023_mystyle++,
    author = {Zeng, Libing and Chen, Lele and Xu, Yi and Kalantari, Nima Khademi},
    title = {MyStyle++: A Controllable Personalized Generative Prior},
    booktitle={ACM SIGGRAPH Asia},
    year={2023}
}
```

## Acknowledgements

This implementation is adapted from [MyStyle](https://github.com/google/mystyle). We gratefully acknowledge the authors for their outstanding work.

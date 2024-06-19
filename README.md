<div align=center>
<h1>  $M^3$-UDA: A New Benchmark for Unsupervised Domain Adaptive Fetal Cardiac Structure Detection.</h1>
</div>
<div align=center>

<!-- <a src="https://img.shields.io/badge/%F0%9F%93%96-ICCV_2023-8A2BE2.svg?style=flat-square" href="https://arxiv.org/abs/2309.11145">
<img src="https://img.shields.io/badge/%F0%9F%93%96-ICCV_2023-8A2BE2.svg?style=flat-square">
</a> -->
   
<a src="https://img.shields.io/badge/%F0%9F%9A%80-xmed_Lab-ed6c00.svg?style=flat-square" href="https://xmengli.github.io/">
<img src="https://img.shields.io/badge/%F0%9F%9A%80-xmed_Lab-ed6c00.svg?style=flat-square">
</a>

<!-- <a src="https://img.shields.io/badge/%F0%9F%9A%80-XiaoweiXu's Github-blue.svg?style=flat-square" href="https://github.com/XiaoweiXu/CardiacUDA-dataset">
<img src="https://img.shields.io/badge/%F0%9F%9A%80-Xiaowei Xu's Github-blue.svg?style=flat-square">
</a> -->

</div>


## :hammer: PostScript
&ensp; :smile: This project is the pytorch implemention of $M^3$-UDA: A New Benchmark for Unsupervised Domain Adaptive Fetal Cardiac Structure Detection;

&ensp; :laughing: Our experimental platform is configured with <u>One *RTX3090 (cuda>=11.0)*</u>; 

&ensp; :blush: Currently, this code is avaliable for proposed dataset FCS and public dataset <u>CardiacUDA</u>;

<!-- &ensp; :smiley: For codes and accessment that related to dataset ***CardiacUDA***; -->

&ensp; &ensp; &ensp;    **:eyes:** The code is now available at:
&ensp; &ensp; &ensp;       ```
                            ..\data\detus_dataset.py
                           ```

<!-- &ensp; :heart_eyes: For codes and accessment that related to dataset ***CardiacUDA*** -->

&ensp; &ensp; &ensp;    **:eyes:** The dataset is coming soon.：


## :computer: Installation


1. You need to build the relevant environment first, please refer to : [**requirements.yaml**](requirements.yaml)

2. Install Environment:
    ```
    conda env create -f requirements.yaml
    ```

+ We recommend you to use Anaconda to establish an independent virtual environment, and python > = 3.8.3; 


## :blue_book: Data Preparation

### *1. FCS dataset*
 * This project provides the use case of Unsupervised Domain Adaptive Fetal Cardiac Structure Detection task;

 * The hyper parameters setting of the dataset can be found in the **utils/config.py**, where you could do the parameters modification;

 * For different tasks, the composition of data sets have significant different, so there is no repetition in this file;


   <!-- #### *1.1. Download The **FCS**.* -->
   <!-- :speech_balloon: The detail of CAMUS, please refer to: https://www.creatis.insa-lyon.fr/Challenge/camus/index.html/. -->

   1. Download & Unzip the dataset.

      The ***CAMUS dataset*** is composed as: /Hospital1 & /Hospital2 & Hospital3.

   2. The source code of loading the FCS dataset exist in path :

      ```python
      ..\data\fetus_dataset.py
      and modify the dataset path in
      ..\utils/config.py
      ```

   <!-- #### *1.2. Download The **CardiacUDA**.*

   :speech_balloon: The detail of CardiacUDA, please refer to: https://echonet.github.io/dynamic/.

   1. Download & Unzip the dataset.

      - The ***CardiacUDA*** dataset is consist of: /Video, FileList.csv & VolumeTracings.csv.

   2. The source code of loading the Echonet dataset exist in path :

      ```python
      ..\datasets\echo.py
      and modify the dataset path in
      ..\train_camus_echo.py
      ``` -->

## :feet: Training

1. In this framework, after the parameters are configured in the file **train_graph_his_debug.py** , you only need to use the command:

    ```shell
    python train_graph_his_debug.py
    ```

2. You are also able to start distributed training. 

   - **Note:** Please set the number of graphics cards you need and their id in parameter **"enable_GPUs_id"**.

#


<!-- ###### :rocket: Code Reference 
  - https://github.com/huawei-noah/Efficient-AI-Backbones/tree/master/vig_pytorch
  - https://github.com/chengchunhsu/EveryPixelMatters 

###### :rocket: Updates Ver 1.0（PyTorch）
###### :rocket: Project Created by Jiewen Yang : jyangcu@connect.ust.hk
======= -->

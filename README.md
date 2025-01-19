# Mixture of Experts for Audio-Visual Learning
This repository is the official implementation of our paper "Mixture of Experts for Audio-Visual Learning", accepted by NeurIPS 2024.


![overview](images/overview.png)
<small>*Method Overview*</small>

In this paper, we explore parameter-efficient transfer learning for audio-visual learning and propose the Audio-Visual Mixture of Experts (AVMOE) to inject adapters into pre-trained models flexibly. Specifically, we introduce unimodal and cross-modal adapters as multiple experts to specialize in intra-modal and inter-modal information, respectively, and employ a lightweight router to dynamically allocate the weights of each expert according to the specific demands of each task. 

For more details, please check the latest version of the paper: [Mixture of Experts for Audio-Visual Learning](https://openreview.net/pdf?id=SNmuKbU0am)

------

### 📝Requirements and Installation

- ###### Getting Started

    ```python
    cd AVMOE
    pip install -r requirements.txt
    ```

- ###### Download HTS-AT Backbone

    Download `checkpoints.zip` from [Baidu Disk](https://pan.baidu.com/s/1oe8beqNiH9bA4geRlHTC7g?pwd=2025) (pwd: 2025), and extract it into the directory `./AVMOE/.`

## AVE
- ###### Download Data

  Download `frames.zip` [Baidu Disk](https://pan.baidu.com/s/1no7R-AJK0A8aQfiFRnAmJQ?pwd=2025) (pwd: 2025), `wave.zip` from [Baidu Disk](https://pan.baidu.com/s/15fKrgbyQmjhZtXY6-d2T0w?pwd=2025) (pwd: 2025), and extract them into the directory `./data/AVE`/.

- ###### Usage

  Go to AVE task directory.

  ```
  cd AVMOE/AVE
  ```

  - Train

    ```c
    bash train.sh
    ```
  - Test

    ```c
    bash test.sh
    ```

## AVQA
- ###### Download Data

  Download `frames.zip` from [Baidu Disk](https://pan.baidu.com/s/1ON8e708Pg_MymgPNJE_fkg?pwd=2025) (pwd: 2025), `audio_wave.zip` from [Baidu Disk](https://pan.baidu.com/s/12SP27Op9Y1rSQUlVaYztpQ?pwd=2025) (pwd: 2025), and extract them into the directory `./data/AVQA/`.

- ###### Usage

  Go to AVQA task directory.

  ```
  cd AVMOE/AVQA
  ```

  - Audio-Visual Grounding Generation

    - Download the `./grounding_gen/models_grounding_gen/lavish_grounding_gen_best.pt` from [Baidu Disk](https://pan.baidu.com/s/10zQcRtnvWgY6jsr0l-oaTA?pwd=2025) (pwd: 2025) to skip the **Audio-Visual Grounding Generation** process.

    - Or, run the below script:
        ```python
        python grounding_gen/main_grd_gen.py
        ```

  - Train

    ```
    bash train_v2.sh
    ```

  - Test
    ```
    bash test_v2.sh
    ```

## AVVP
- ###### Download Data

  Download extracted feats, frame and wave of LLP dataset from [Baidu Disk](https://pan.baidu.com/s/1k9dXRTmub0SeSebozIryfw?pwd=2025) (pwd: 2025), and extract it into the directory `./data/AVVP/`. 

- ###### Usage

  Go to AVVP task directory:

  ```
  cd AVMOE/AVVP
  ```

  - Train

    ```
    bash train.sh
    ```

  - Test

    ```
    bash test.sh
    ```
## AVS
- ###### Download Data

  - Download Dataset

    The updated AVSBench dataset is available [here](http://www.avlbench.opennlplab.cn/download) (`AVSBench-object`). You may request the dataset by filling the [Google Form](https://forms.gle/GKzkU2pEkh8aQVHN6).

    The downloaded data should be placed to the directory `./data/`.

  - Download Wave

    Download wave for task **S4** ([Baidu Disk](https://pan.baidu.com/s/1acWu8o6PfoYz8cPywRoVog?pwd=2025) (pwd: 2025)) and task **MS3** ([Baidu Disk](https://pan.baidu.com/s/1ZTrhWKTyDgS0zXLFtapIVg?pwd=2025) (pwd: 2025)), and extract them into the directory `./data/AVSBench_data/Single-source/s4_data/` and `./data/AVSBench_data/Multi-sources/ms3_data/`, respectively.

- ###### Download pretrained backbones

  The pretrained ResNet50/PVT-v2-b5 (vision) and VGGish (audio) backbones can be downloaded from [Baidu Disk](https://pan.baidu.com/s/1vtcnqPU6mzyQz1okDkXS5w?pwd=2025) (pwd: 2025) and placed to the directory `./AVMOE/AVS/pretrained_backbones/`.

- ###### Usage

  Go to AVS task directory.

  ```python
  # for S4 task:
  cd AVMOE/AVS/avs_scripts/avs_s4
  
  # for MS3 task:
  cd AVMOE/AVS/avs_scripts/avs_ms3
  ```

  - Train

    ```python
    bash train_v2.sh
    ```

  - Test

    ```
    bash test_v2.sh
    ```

## 👍Acknowledgments

Our code is based on [DG-SCT](https://github.com/haoyi-duan/DG-SCT), [CMBS](https://github.com/marmot-xy/CMBS), [AVSBench](https://github.com/OpenNLPLab/AV[SBench), [MGN](https://github.com/stoneMo/MGN), [MUSIC-AVQA](https://github.com/GeWu-Lab/MUSIC-AVQA), and [LAVisH](https://github.com/GenjiB/LAVISH).


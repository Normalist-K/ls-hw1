# Instructions for HW1 (Youngin Kim, youngin2)
HW1: 11-775 Large-Scale Multimedia Analysis, Spring 2022

## 0. Prerquisite

**Please modify `config.sh` for your own path**
- `BASE_DIR` : Base directory path that your *codes* are saved. 
- `DATA_DIR` : Data directory path that your *data* are saved.

Dependencies: FFMPEG, Python: sklearn, pandas
```
# [OPTIONAL] create conda environment
$ conda create -n myenv python=3.8
$ conda activate myenv

# install FFMPEG
$ apt-get install ffmpeg

# install pytorch according to instructions
# https://pytorch.org/get-started/
# ex) conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch

# install requirements
$ pip install -r requirements.txt
```
Dependencies: OpenSMILE
```
$ bash sh/install_opensmile.sh
```
## 1. Data and Labels

1. Download video data: same as baseline code TA provided.
- [download link](https://drive.google.com/file/d/1WEINPdvQ1ZUELxaXlhHcvoOjEML8gYYY/view?usp=sharing)

2. Uncompress the data into *the folder you use*. 
3. split data to K-Fold
```
$ bash sh/split.sh
```


4. Extract the audios(.wav & .mp3) from the videos
- .wav: for MFCC-Bag-Of-Feature
- .mp3: for SoundNet-Global-Pool (sampling rate: 22050)
```
$ bash sh/extract_audio.sh
```

## 2. Feature Extractor

1. MFCC-Bag-Of-Feature
```
$ bash sh/mfcc.sh
```
2. SoundNet-Global-Pool
    -  reference: https://github.com/salmedina/soundnet_pytorch
```
$ bash sh/soundnet.sh
```

## 3. Classifier results

1. mfcc.csv : MFCC-Bag-Of-Feature + SVM classifier
```
$ bash sh/mfcc_svm.sh
```

2. soundnet.csv : SoundNet(y_scns) + MLP classifier
```
$ bash sh/soundnet_mlp.sh
```

3. best.csv : AST(Audio Spectrogram Transformer) + Linear head
    - reference: https://github.com/YuanGongND/ast
```
$ bash sh/best.sh
```
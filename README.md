# Speech-Emotion-Recognition-System

## Description
Speech-Emotion-Recognition-System is a project that can detect your emotion with few seconds recording. In this project, we choose [EmoDB](http://www.emodb.bilderbar.info/navi.html) and [RAVDESS](https://zenodo.org/record/1188976#.XTBWj-gzZEZ) as the training corpus.

## Environment: 
- OS: Ubuntu 16.04 LTS
- Programming language: python 2.7
## GUI
![](https://i.imgur.com/APnRqHK.jpg)
## Implementation
Use .wav files from EmoDB and RAVDESS as inputs
- **Preprocessing**:
  - Use voice activation detection to get the voiced signal.
  - Pre-emphasis
  - Framing
  - Windowing
  
- **Feature extraction**: Extract acoustic features (e.g. Pitch, Energy, MFCC, etc.) from preprocessed audio signal.

- **Create dataset**: Use the extracted features of audio signals to create a csv format dataset.

- **Train**: Use SVM classification machine learning algorithm to train our model.

- **Features of GUI**:
  - Input: Record or load a .wav file
  - Feature extraction: Use matplotlib to visualize extracted features.
  - Classify: Use pretrained model to identify the emotion of the input signal.
  - Music recommendation: According to your emotion, we will recommend some music corresponding to your mood. 
  
## What I learned
1. Know how to preprocess audio signal.
  - Packages I used:
    - [numpy](https://github.com/numpy/numpy)
    - [webrtcvad](https://github.com/wiseman/py-webrtcvad)
2. Know how to extract acoustic features from audio signal.
  - Packages I used:
    - [numpy](https://github.com/numpy/numpy)
    - [scipy](https://github.com/scipy/scipy)
3. Know how to create a dataframe and write csv file.
  - Packages I used:
    - [pandas](https://github.com/pandas-dev/pandas)
4. Know the basic concept of SVM and build an SVM model.
  - Packages I used:
    - [scikit-learn](https://github.com/scikit-learn/scikit-learn)
5. Know how to build a desktop application.
  - Packages I used:
    - PyQt5
    - [matplotlib](https://github.com/matplotlib/matplotlib)
    - [spotipy](https://github.com/plamere/spotipy) - Spotify API python interface, we connect spotify API for music recommendation.
    - urllib

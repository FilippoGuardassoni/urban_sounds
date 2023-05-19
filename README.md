![](https://github.com/FilippoGuardassoni/urban_sounds/blob/main/img/headerheader.jpg)

# Urban Sounds Classification

![GitHub release (latest by date including pre-releases)](https://img.shields.io/github/v/release/pragyy/datascience-readme-template?include_prereleases)
![GitHub last commit](https://img.shields.io/github/last-commit/FilippoGuardassoni/spotify_hitsong)
![GitHub pull requests](https://img.shields.io/github/issues-pr/FilippoGuardassoni/spotify_hitsong)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
![contributors](https://img.shields.io/github/contributors/FilippoGuardassoni/spotify_hitsong) 
![codesize](https://img.shields.io/github/languages/code-size/FilippoGuardassoni/spotify_hitsong)

# Project Overview

Every day we hear different sounds from different sources. We can recognize several random examples of sounds from the environment such as an ambulance, an engine or an airplane. But can we train a computer to classify these random sounds? If the answer is yes, it could be used for surveillance and monitoring or for other environmental researches such as the urban sound pollution in the cities, and it can be applied to the general audio recognition as well. The objective of this paper is to create a suited machine learning technique to classify urban sounds accurately. The dataset used is the UrbanSound8K audio dataset, comprised of 8732 audio clips of urban sounds divided in 10 classes. Two different approaches are followed: one using Artificial Neural Network (ANN) fed with raw data of extracted features of sound and one using Convolutional Neural Network (CNN) fed with images of the spectrums of sound. After testing the models with 5-fold cross validation, the algorithm that performed the best was the Artificial Neural Network, when paired with the use of 40 Mel-Frequency Cepstral Coefficients as input features.

# Installation and Setup

- Google Colab

## Codes and Resources Used
- Python 2.7 and up

## Python Packages Used
- **General Purpose:** General purpose packages like `urllib, os, request`, and many more.
- **Data Manipulation:** Packages used for handling and importing dataset such as `pandas, numpy` and others.
- **Machine Learning:** This includes packages that were used to generate the ML model such as `scikit, tensorflow`, etc.

# Data
The UrbanSound8k contains 8732 audio files in the format .wav divided in 10 classes, each one representing a different type of city sound, for instance, we can find car horns, dogs barking, sirens, etc.

## Source Data & Data Acquisition
The UrbanSound8k dataset can be downloaded at https://urbansounddataset.weebly.com/urbansound8k.html

## Data Pre-processing
Since the ranges among the feature vectors components vary too much, standardization with the StandardScaler function from scikit-learn package is applied. This function centered the dataset with mean approximately 0 and standard deviation 1.
The labels which represent the class of sounds are encoded from categorical to numerical data type.
As a further step, Principal Component Analysis (PCA) is performed to have an additional perspective. From 114 features, the dimensions were reduced to 90 with the 90% variance criterium.

## Image Pre-processing
An image has three or one channel: red, green and blue, or grey. Image classification is a time-consuming task; therefore, images are converted to greyscale in order to reduce the weight. In particular, the use of the Librosa library in combination with the cv2 library allowed the transformation of the spectrum images such as scaling pixel values to prevent too wide ranges and resizing in order to have same length arrays (necessary to feed the neural network).

# Project structure

```bash
├── dataset
│   ├── features.csv
├── code
│   ├── urbansoundclassification.py
├── report
│   ├── urban_sound_classification_report.pdf
├── img
│   ├── headerheader.jpg      
├── LICENSE
└── README.md
```

# Results and evaluation
The test sets are composed by the folds within the UrbanSound8k that are fold 5, fold 7, fold 8, fold 9 and fold 10. The test sets for both types of neural networks are pre-processed following respectively the same procedures of the training sets.
The MLP with Dropout Model performed better for the classification. The average accuracy and standard deviation of MLP and CNN model is as follows:

![image](https://github.com/FilippoGuardassoni/urban_sounds/assets/85356795/6fe43d2c-0cb9-4f0c-95cd-f1a02b757bbe)


# Future work
There are two main limitations in this paper. The first one being what features of sound are considered and their variations. This limitation is proportional to the knowledge of the field. The other limitation is attributed to the tools used to conduct the research, which have limited capabilities in a limited time window, especially when it comes to train CNN models (usually trained for weeks).
Regarding the dataset, we considered the most popular features along their mean and standard deviation, however more features such as Tonnetz and for example the minimum and the maximum could have been considered. This is also directly related to Data Augmentation: it helps to generate synthetic data from existing data set such that generalization capability of model can be improved. Techniques such as noise injection, shifting time, changing pitch and speed should be implemented to have a more complete dataset in order to interpret correctly sounds directly from the environment. In fact, it should be taken into consideration that the dataset itself is imperfect because the audio files derive from different sources with different qualities (due to the digitalization). Regarding the machine learning part of the research, more computational power combined with a proper knowledge of the field would allow to construct a more suited and complex architecture for CNN, which should be trained for the proper amount of time. This will also allow to consider images with more channels (colors) and bigger size for increased learning capacity. A different approach with other types of neural networks such as Recurrent Neural Network could be also attempted. Additionally, we implemented PCA for dimensionality reduction, but it might not be indicated as the achieved performances are limited. Instead, other feature selection techniques could be attempted


# Acknowledgments/References
See report/song_hit_prediction.pdf for references.

# License
For this github repository, the License used is [MIT License](https://opensource.org/license/mit/).

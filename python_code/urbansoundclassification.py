# Importing Data, Libraries and First Look

## Import
"""

from google.colab import drive
drive.mount('/content/drive/')

import sys
sys.path.insert(0,'/content/drive/MyDrive/dataset')

import tarfile 
tar = tarfile.open('/content/drive/MyDrive/dataset/UrbanSound8K.tar.gz')
tar.extractall()
tar.close()

"""## Libraries"""

# Commented out IPython magic to ensure Python compatibility.
# Required libraries
import os
import pandas as pd
import IPython as IP
import struct
import matplotlib.pyplot as plt
# %matplotlib inline
import numpy as np
import seaborn as sns
import librosa
import librosa.display
import pickle
import pathlib
import csv
import soundfile as sf
!pip install pydub
from pydub import AudioSegment
import sklearn
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from PIL import Image
import pickle
import time
import struct

"""## Dataset"""

# Set your path to the original dataset
us8k_path = os.path.abspath('/content/UrbanSound8K' )

# Global settings
metadata_path = os.path.join(us8k_path, '/content/UrbanSound8K/metadata/UrbanSound8K.csv')
audio_path = os.path.join(us8k_path, '/content/UrbanSound8K/audio')

print("Loading CSV file {}".format(metadata_path))

# Load metadata as a Pandas dataframe
metadata = pd.read_csv(metadata_path)

# Examine dataframe's head
metadata.head()

# Group-by folds
fold_count = metadata['fold'].value_counts()
fold_count

"""# Exploratory Data Analysis (EDA)

## MV, Duplicates, Balance
"""

# Check missing values in Metadata file
metadata.isnull().values.any()

# Checking Duplicate Files in the Metadata File
metadata.duplicated(subset=['slice_file_name']).any()

# Checking the dataset is balanced or not 

plt.figure(figsize = (8,5))
sns.set_theme(style="whitegrid")
ax = sns.countplot(x="class", data=metadata, palette="RdBu")
ax.set_xticklabels(ax.get_xticklabels(),rotation = 90)

# Class distribution
metadata['class'].value_counts()

"""## Audio Properties"""

def read_header(filename):
    wave = open(filename,"rb")
    riff = wave.read(12)
    fmat = wave.read(36)
    num_channels_string = fmat[10:12]
    num_channels = struct.unpack('<H', num_channels_string)[0]
    sample_rate_string = fmat[12:16]
    sample_rate = struct.unpack("<I",sample_rate_string)[0]
    bit_depth_string = fmat[22:24]
    bit_depth = struct.unpack("<H",bit_depth_string)[0]
    return (num_channels, sample_rate, bit_depth)

# Read every file header to collect audio properties
audiodata = []
for index, row in metadata.iterrows():
    id = index
    cat = str(row["class"])
    fold = 'fold'+str(row["fold"])
    name = str(row["slice_file_name"])
    file_name = os.path.join(audio_path, fold, name)
    audio_props = read_header(file_name)
    duration = row['end'] - row['start']
    audiodata.append((id, name, fold, cat, duration) + audio_props)

# Convert into a Pandas dataframe
audiodatadf = pd.DataFrame(audiodata, columns=['id', 'filename', 'fold', 'class', 'duration', 'channels','sample_rate','bit_depth'])

audiodatadf

audiodatadf.describe()

# Audio Lengths (duration)
# Plot audio lengths distribution
plt.hist(audiodatadf['duration'], rwidth=0.9, color='#86bf91')

plt.xlabel('Duration')
plt.ylabel('Population')
plt.title('Histogram of audio lengths')
plt.grid(False)
plt.show()

# Count samples with duration > 3 sec
gt_3sec = audiodatadf['duration'][audiodatadf['duration'] > 3].count()
lt_3sec = audiodatadf['duration'][audiodatadf['duration'] < 3].count()
lt_15sec = audiodatadf['duration'][audiodatadf['duration'] < 1.5].count()

# Display counts of interest
print("Greater than 3 seconds: {}".format(gt_3sec))
print("Lower than 3 seconds: {}".format(lt_3sec))
print("Lower than 1.5 seconds: {}".format(lt_15sec))

# Audio Channels
print(audiodatadf.channels.value_counts(normalize=True))

# Bit Depths
print("Bit depths:\n")
print(audiodatadf.bit_depth.value_counts(normalize=True))

# Sample Rates
print("Sample rates:\n")
print(audiodatadf.sample_rate.value_counts(normalize=True))

# Plot wavesounds for each audio class
# Getting a random audio file for each class
np.random.seed(0)
random_class_df = pd.DataFrame(audiodatadf.groupby('class')['id'].apply(np.random.choice).reset_index())
random_class_df

random_class_merge = pd.merge(left = audiodatadf, right = random_class_df, left_on = 'id', right_on = 'id')
random_class_merge

# Reading data for the random audio files selected
random_class_data = []

for idx in random_class_merge.index:  
    wav, sr = librosa.load(audio_path + '/' + str(random_class_merge['fold'][idx]) + '/' + str(random_class_merge['filename'][idx]))
    random_class_data.append(wav)

# Plotting the waveforms for each class
fig, ax = plt.subplots(nrows = 5, ncols = 2, figsize = (16, 30))
for i in range(5):
    librosa.display.waveplot(random_class_data[2*i], sr = sr, ax = ax[i][0])
    ax[i][0].set_title(random_class_merge['class_x'][2*i])
    
    librosa.display.waveplot(random_class_data[2*i + 1], sr = sr, ax = ax[i][1])
    ax[i][1].set_title(random_class_merge['class_x'][2*i + 1])
    
plt.show()

"""# Features

## Overview

### Short-Time Fourier Transform (STFT)
"""

# Selecting a random File 
row = metadata.sample(1)
file_path = audio_path + '/fold'+ str(row.iloc[0,5]) +'/' + str(row.iloc[0,0])

# Windowing
n_fft=2048
hop_length=512

# Load audio file
y, sr = librosa.load(file_path)

# Normalize between -1 and 1
normalized_y = librosa.util.normalize(y)

# Compute STFT
stft = librosa.core.stft(normalized_y, n_fft=n_fft, hop_length=hop_length)

# Convert sound intensity to log amplitude:
stft_db = librosa.amplitude_to_db(abs(stft))

# Plot spectrogram from STFT
plt.figure(figsize=(12, 4))
librosa.display.specshow(stft_db, x_axis='time', y_axis='log')
plt.colorbar(format='%+2.0f dB');
plt.title('STFT Spectrogram')
plt.tight_layout()
plt.show()

"""### Mel Frequency Cepstral Coefficients (MFCCs)"""

# Generate MFCC coefficients
mfcc = librosa.feature.mfcc(normalized_y, sr, n_mfcc=40)

# Plot spectrogram from STFT
plt.figure(figsize=(12, 4))
librosa.display.specshow(mfcc, x_axis='time', y_axis='mel')
plt.colorbar()
plt.title('MFCC')
plt.tight_layout()
plt.show()

"""### Mel Scaled Spectrogram"""

#Mel scaled Filter Banks
n_mels = 128

# Generate mel scaled spectrogram
mel = librosa.feature.melspectrogram(S=stft, n_mels=n_mels)

# Convert sound intensity to log amplitude:
mel_db = librosa.amplitude_to_db(abs(mel))

# Normalize between -1 and 1
normalized_mel = librosa.util.normalize(mel_db)

# Plot spectrogram from STFT
plt.figure(figsize=(12, 4))
librosa.display.specshow(mel_db, x_axis='time', y_axis='mel')
plt.colorbar(format='%+2.0f dB');
plt.title('MEL-Scaled Spectrogram')
plt.tight_layout()
plt.show()

"""### Root-Mean-Square (RMS)"""

# Get RMS value from each frame's magnitude value
S, phase = librosa.magphase(librosa.stft(normalized_y))
rms = librosa.feature.rms(S=S)

# Plot the RMS energy
fig, ax = plt.subplots(figsize=(15, 6), nrows=2, sharex=True)
times = librosa.times_like(rms)
ax[0].semilogy(times, rms[0], label='RMS Energy')
ax[0].set(xticks=[])
ax[0].legend()
ax[0].label_outer()
plt.title('RMS')
librosa.display.specshow(librosa.amplitude_to_db(S, ref=np.max),
                         y_axis='log', x_axis='time', ax=ax[1])
ax[1].set(title='log Power spectrogram')

"""### Zero Crossing Rate (ZCR)"""

# Get ZCR value from each frame
zcrs = librosa.feature.zero_crossing_rate(normalized_y)
print(f"Zero crossing rate: {sum(librosa.zero_crossings(normalized_y))}")

#Plotting ZCR
plt.figure(figsize=(15, 3))
plt.plot(zcrs[0])
plt.title('ZCR')

"""### Chroma (CHROMA)"""

# Get CHROMA value
hop_length = 512
chromagram = librosa.feature.chroma_stft(normalized_y, sr=sr, hop_length=hop_length)

# Plotting the CHROMA
fig, ax = plt.subplots(figsize=(15, 3))
img = librosa.display.specshow(chromagram, x_axis='time', y_axis='chroma', hop_length=hop_length, cmap='coolwarm')
plt.title('CHROMA')
fig.colorbar(img, ax=ax)

"""### Spectral Centroid (SC) and Spectral Bandwidth (SB)"""

# Spectral centroid
spectral_centroids = librosa.feature.spectral_centroid(normalized_y, sr=sr)[0]
spectral_centroids.shape 

# Computing the time variable for visualization
frames = range(len(spectral_centroids))
t = librosa.frames_to_time(frames)

# Normalising the spectral centroid for visualisation
def normalize(x, axis=0):
    return sklearn.preprocessing.minmax_scale(x, axis=axis)

# Plotting the Spectral Centroid along the waveform
librosa.display.waveplot(normalized_y, sr=sr, alpha=0.4)
plt.plot(t, normalize(spectral_centroids), color='r')
plt.title('SC')

# Spectral Bandwidth
spectral_bandwidth = librosa.feature.spectral_bandwidth(normalized_y, sr=sr)[0]
spectral_bandwidth.shape

# Computing the time variable for visualization
frames = range(len(spectral_bandwidth))
t = librosa.frames_to_time(frames)

# Plotting the Spectral Bandwidth along the waveform
librosa.display.waveplot(normalized_y, sr=sr, alpha=0.4)
plt.plot(t, normalize(spectral_bandwidth), color='r')
plt.title('SB')

"""### Tempogram"""

hop_length = 512

# Compute local onset autocorrelation
oenv = librosa.onset.onset_strength(y=normalized_y, sr=sr, hop_length=hop_length)
times = librosa.times_like(oenv, sr=sr, hop_length=hop_length)
tempogram = librosa.feature.tempogram(onset_envelope=oenv, sr=sr,
                                      hop_length=hop_length)
# Estimate the global tempo for display purposes
tempo = librosa.beat.tempo(onset_envelope=oenv, sr=sr,
                           hop_length=hop_length)[0]
# Plotting the tempogram
fig, ax = plt.subplots(figsize=(15, 3))
img = librosa.display.specshow(tempogram, x_axis='time', y_axis='tempo', hop_length=hop_length, cmap='coolwarm')
fig.colorbar(img, ax=ax)

"""## Audio Pre-Processing"""

# Modify different encoders to read audio files with pydub

folds = 'fold1 fold10 fold2 fold3 fold4 fold5 fold6 fold7 fold8 fold9'.split()
for f in folds:
     pathlib.Path(f'{us8k_path}/new_audio/{f}').mkdir(parents=True, exist_ok=True)
     for filename in os.listdir(audio_path + str('/') + str(f)):
        audioname = f'{audio_path}/{f}/{filename}'
        if filename == '.DS_Store':
          continue
        k = sf.SoundFile(audioname)
        if k.subtype != "PCM_32":
          file, samplerate = sf.read(audioname)
          sf.write(f'{us8k_path}/new_audio/{f}/{filename}', file, samplerate, 'PCM_32')
        else:
          audioname.export(f'{us8k_path}/new_audio/{f}/{filename}', format='wav')

# Define new audio path
new_audio_path = os.path.join(us8k_path, '/content/UrbanSound8K/new_audio')

"""### Converting To Two Channels"""

# Coverting audio channels from mono to stereo

folds = 'fold1 fold10 fold2 fold3 fold4 fold5 fold6 fold7 fold8 fold9'.split()
for f in folds:
     pathlib.Path(f'{us8k_path}/stereo_audio/{f}').mkdir(parents=True, exist_ok=True)
     for filename in os.listdir(new_audio_path + str('/') + str(f)):
        audioname = f'{new_audio_path}/{f}/{filename}'
        if filename == '.DS_Store':
          continue
        else:
          audio = AudioSegment.from_wav(audioname)
          audio = audio.set_channels(2)
          audio.export(f'{us8k_path}/stereo_audio/{f}/{filename}', format='wav')

# Define new audio path
stereo_audio_path = os.path.join(us8k_path, '/content/UrbanSound8K/stereo_audio')

"""### Resize Length"""

# Run to remove folder
# Used in debugging
import shutil

shutil.rmtree('/content/UrbanSound8K/padded_audio')

# Padding audio files to the same length
# Define fixed length (in milliseconds)
pad_ms = 4037 

folds = 'fold1 fold10 fold2 fold3 fold4 fold5 fold6 fold7 fold8 fold9'.split()
for f in folds:
     pathlib.Path(f'{us8k_path}/padded_audio/{f}').mkdir(parents=True, exist_ok=True)
     for filename in os.listdir(stereo_audio_path + str('/') + str(f)):
        audioname = f'{stereo_audio_path}/{f}/{filename}'
        audio = AudioSegment.from_wav(audioname)
        audio = AudioSegment.from_file(audioname, format="wav", duration=round(len(audio),3))
        if pad_ms == len(audio):
          audio.export(f'{us8k_path}/padded_audio/{f}/{filename}', format='wav')
          print("Original Audio: " + str(len(audio)))
        else:
          silence = AudioSegment.silent(duration=pad_ms-len(audio))
          padded = audio + silence  # Adding silence after the audio
          padded.export(f'{us8k_path}/padded_audio/{f}/{filename}', format='wav') # Exporting to folder
          print("Original Audio: " + str(len(audio)) + "  " + "Padded Audio: " + str(len(padded)))

# Define new audio path
padded_audio_path = os.path.join(us8k_path, '/content/UrbanSound8K/padded_audio')

# Store new information into dataframe
paddedaudio = []
folds = 'fold1 fold10 fold2 fold3 fold4 fold5 fold6 fold7 fold8 fold9'.split()
for f in folds:
     for filename in os.listdir(padded_audio_path + str('/') + str(f)):
        audioname = f'{padded_audio_path}/{f}/{filename}'
        audio = AudioSegment.from_wav(audioname)
        duration = len(audio)
        paddedaudio.append((filename, duration))


paddedaudiodf = pd.DataFrame(paddedaudio, columns=['filename', 'duration'])

# Check audio have same length
paddedaudiodf.describe()

# Save zipped padded audio in Drive
!zip -r '/content/drive/MyDrive/dataset/padded_audio.zip' '/content/UrbanSound8K/padded_audio'

"""## Extraction

### Entire Feature Set
"""

# Run to remove folder
# Used in debugging
import shutil

shutil.rmtree('/content/UrbanSound8K/padded_audio')

# Convert the audio data files into PNG format images
cmap = plt.get_cmap('inferno')
plt.figure(figsize=(8,8))
folds = 'fold1 fold10 fold2 fold3 fold4 fold5 fold6 fold7 fold8 fold9'.split()
for f in folds:
    pathlib.Path(f'{us8k_path}/png/{f}').mkdir(parents=True, exist_ok=True)
    for filename in os.listdir(padded_audio_path + str('/') + str(f)):
        audioname = f'{padded_audio_path}/{f}/{filename}'
        y, sr = librosa.load(audioname)
        plt.specgram(y, NFFT=2048, Fs=2, Fc=0, noverlap=128, cmap=cmap, sides='default', mode='default', scale='dB');
        plt.axis('off');
        plt.savefig(f'{us8k_path}/png/{f}/{filename[:-4]}.png', format = 'png')
        plt.clf()

# Check file number to match original
folds = 'fold1 fold10 fold2 fold3 fold4 fold5 fold6 fold7 fold8 fold9'.split()
for f in folds:
  png_count = len(os.listdir("/content/UrbanSound8K/png" + str('/') + str(f)))
  print(str(f) + ": " + str(png_count))

# Save zipped png folder to drive
!zip -r '/content/drive/MyDrive/dataset/png.zip' '/content/UrbanSound8K/png'

# Restore zipped padded audio folder to drive
!unzip '/content/drive/MyDrive/dataset/padded_audio.zip' -d '/'

# Define new audio path
padded_audio_path = os.path.join(us8k_path, '/content/UrbanSound8K/padded_audio')

# Create a header for our CSV file
header = 'filename rmse_avg rmse_std spectral_centroid_avg spectral_centroid_std spectral_bandwidth_avg spectral_bandwidth_std rolloff_avg rolloff_std zero_crossing_rate_avg zero_crossing_rate_std'
for i in range(1, 13):
    header += f' chroma_stft_avg{i} chroma_stft_std{i}'
for i in range(1, 41):
    header += f' mfcc_avg{i} mfcc_std{i}'
header += ' label'
header = header.split()

# Extract the spectrogram for every audio
file = open('/content/UrbanSound8K/features.csv', 'w', newline='')
with file:
    writer = csv.writer(file)
    writer.writerow(header)
folds = 'fold1 fold10 fold2 fold3 fold4 fold5 fold6 fold7 fold8 fold9'.split()
for f in folds:
    for filename in os.listdir(padded_audio_path + str('/') + str(f)):
        audioname = f'{padded_audio_path}/{f}/{filename}'
        y, sr = librosa.load(audioname)
        rms = librosa.feature.rms(y=y)
        chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
        spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)
        spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
        rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
        zcr = librosa.feature.zero_crossing_rate(y)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
        to_append = f'{filename} {np.mean(rms)} {np.std(rms)} {np.mean(spec_cent)} {np.std(spec_cent)} {np.mean(spec_bw)} {np.std(spec_bw)} {np.mean(rolloff)} {np.std(rolloff)} {np.mean(zcr)} {np.std(zcr)}'
        for e in chroma_stft:
            to_append += f' {np.mean(e)} {np.std(e)}'  
        for e in mfcc:
            to_append += f' {np.mean(e)} {np.std(e)}'
        to_append += f' {f}'
        file = open('/content/UrbanSound8K/features.csv', 'a', newline='')
        with file:
            writer = csv.writer(file)
            writer.writerow(to_append.split())

#Export csv to Drive
!cp /content/UrbanSound8K/features.csv '/content/drive/MyDrive/dataset'

"""# Data Pre-Processing

## Loading, Dataframe and Summary Stat
"""

pd.set_option('display.max_columns', None)

# Read the data from Drive
data = pd.read_csv('/content/drive/MyDrive/dataset/features.csv')
data.head()

# Add columns from metadata file
features = pd.DataFrame(data)
features = pd.merge(left = audiodatadf, right = features, left_on = 'filename', right_on = 'filename')
features

# Drop unnecessary columns
features = features.drop(features.columns[[0,1, 4, 5, 6, 7, 122]], axis = 1)
features

features.describe()

"""## Training and Test Split"""

# Create training dataset
s = ['fold1','fold2','fold3','fold4','fold6']

features_training = features.loc[features['fold'].isin(s)]
features_training = features_training.drop(features_training.columns[0], axis = 1)
features_training

# Create test dataset Fold 5
features_test_5 = features.loc[features['fold'] =='fold5']
features_test_5 = features_test_5.drop(features_test_5.columns[0], axis = 1)
features_test_5

# Create test dataset Fold 7
features_test_7 = features.loc[features['fold'] =='fold7']
features_test_7 = features_test_7.drop(features_test_7.columns[0], axis = 1)
##
# Create test dataset Fold 8
features_test_8 = features.loc[features['fold'] =='fold8']
features_test_8 = features_test_8.drop(features_test_8.columns[0], axis = 1)

##
# Create test dataset Fold 9
features_test_9 = features.loc[features['fold'] =='fold9']
features_test_9 = features_test_9.drop(features_test_9.columns[0], axis = 1)

##
# Create test dataset Fold 10
features_test_10 = features.loc[features['fold'] =='fold10']
features_test_10 = features_test_10.drop(features_test_10.columns[0], axis = 1)

"""## Scaling"""

# Scale the Training data
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

features_training.iloc[:, 1:] = scaler.fit_transform(features_training.iloc[:, 1:])
features_training

features_training.describe()

scaler = StandardScaler()

features_test_5.iloc[:, 1:] = scaler.fit_transform(features_test_5.iloc[:, 1:])
features_test_7.iloc[:, 1:] = scaler.fit_transform(features_test_7.iloc[:, 1:])
features_test_8.iloc[:, 1:] = scaler.fit_transform(features_test_8.iloc[:, 1:])
features_test_9.iloc[:, 1:] = scaler.fit_transform(features_test_9.iloc[:, 1:])
features_test_10.iloc[:, 1:] = scaler.fit_transform(features_test_10.iloc[:, 1:])

features_test_5.describe()

"""## Class Encoding"""

# Encoding training dataset labels
from sklearn.preprocessing import LabelEncoder

# Create an object of the label encoder classerd
labelencoder = LabelEncoder()

features_training['class'] = labelencoder.fit_transform(features_training['class'])
features_training

##Encoding Test Dataset Labels
features_test_5['class'] = labelencoder.fit_transform(features_test_5['class'])
features_test_7['class'] = labelencoder.fit_transform(features_test_7['class'])
features_test_8['class'] = labelencoder.fit_transform(features_test_8['class'])
features_test_9['class'] = labelencoder.fit_transform(features_test_9['class'])
features_test_10['class'] = labelencoder.fit_transform(features_test_10['class'])

features_test_5

"""# Feature Selection

## Target and Features Split
"""

# Drop Class column and create Independent and Dependent variable datasets for training
X_train = features_training.drop('class', axis=1)

y_train = features_training['class']

# Save target and features for training to drive
X_train.to_csv('/content/drive/MyDrive/dataset/variables/original/x_train.csv',index=False, header=None)
y_train.to_csv('/content/drive/MyDrive/dataset/variables/original/y_train.csv', index=False,header=None)

# Drop Class column and create Independent and Dependent variable datasets for testing
X_test_5 = features_test_5.drop('class', axis=1)
y_test_5 = features_test_5['class']

##
X_test_7 = features_test_7.drop('class', axis=1)
y_test_7 = features_test_7['class']
##
X_test_8 = features_test_8.drop('class', axis=1)
y_test_8 = features_test_8['class']
##
X_test_9 = features_test_9.drop('class', axis=1)
y_test_9 = features_test_9['class']
##
X_test_10 = features_test_10.drop('class', axis=1)
y_test_10 = features_test_10['class']

# Save target and features for test to drive
X_test_5.to_csv('/content/drive/MyDrive/dataset/variables/original/x_test_5.csv', index=False, header=None)
y_test_5.to_csv('/content/drive/MyDrive/dataset/variables/original/y_test_5.csv', index=False,header=None)
##
X_test_7.to_csv('/content/drive/MyDrive/dataset/variables/original/x_test_7.csv', index=False,header=None)
y_test_7.to_csv('/content/drive/MyDrive/dataset/variables/original/y_test_7.csv', index=False,header=None)
##
X_test_8.to_csv('/content/drive/MyDrive/dataset/variables/original/x_test_8.csv', index=False,header=None)
y_test_8.to_csv('/content/drive/MyDrive/dataset/variables/original/y_test_8.csv', index=False,header=None)
##
X_test_9.to_csv('/content/drive/MyDrive/dataset/variables/original/x_test_9.csv', index=False,header=None)
y_test_9.to_csv('/content/drive/MyDrive/dataset/variables/original/y_test_9.csv', index=False,header=None)
##
X_test_10.to_csv('/content/drive/MyDrive/dataset/variables/original/x_test_10.csv', index=False,header=None)
y_test_10.to_csv('/content/drive/MyDrive/dataset/variables/original/y_test_10.csv', index=False,header=None)

type(X_test_5)

"""## Correlation Matrix - Correlation"""

# Compute the correlation matrix
corr = X_train.corr()

# Generate a mask for the upper triangle
mask = np.triu(np.ones_like(corr, dtype=bool))

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(20, 15))

# Generate a custom diverging colormap
cmap = sns.diverging_palette(240, 10, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr, mask=mask, cmap=cmap, vmin=-1, vmax=1, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})

"""## Principal Component Analysis (PCA) - Importance

### Training Dataset
"""

# Reload features for training & test to drive
X_train = pd.read_csv('/content/drive/MyDrive/dataset/variables/original/x_train.csv', header=None)

###
X_test_5 = pd.read_csv('/content/drive/MyDrive/dataset/variables/original/x_test_5.csv', header=None)

##
X_test_7 = pd.read_csv('/content/drive/MyDrive/dataset/variables/original/x_test_7.csv',header=None)

#
X_test_8 = pd.read_csv('/content/drive/MyDrive/dataset/variables/original/x_test_8.csv',header=None)

##
X_test_9 = pd.read_csv('/content/drive/MyDrive/dataset/variables/original/x_test_9.csv',header=None)

##
X_test_10 = pd.read_csv('/content/drive/MyDrive/dataset/variables/original/x_test_10.csv',header=None)
#X_test_10 = X_test_10.to_numpy()

type(X_test_5)

X_test_5.head(10)

X_test_5.shape

print(X_test_5)

# Implement PCA
from sklearn.decomposition import PCA
pca = PCA()
df_pca = pd.DataFrame(pca.fit_transform(X_train))

# Extract the explained variance
explained_variance = pca.explained_variance_ratio_
singular_values = pca.singular_values_

df_pca

# Create an x for each component
x = np.arange(1,len(explained_variance)+1)

# Plot performance
plt.bar(x, explained_variance, alpha=0.5, align='center', label='individual explained variance')
plt.step(range(1,115), np.cumsum(explained_variance), where= 'mid', label='cumulative explained variance')
plt.rcParams["figure.figsize"] = (15, 8)
plt.ylabel('Explained variance ratio')
plt.xlabel('Principal components')
plt.legend(loc = 'best')
plt.show()

#iterate over the components to print the explained variance
total_variance = 0
for i in range(0, 114):
    total_variance += explained_variance[i]
    print(f"Component {i:>2} accounts for {explained_variance[i]*100:>2.2f}% of variance and total is {total_variance*100:>2.2f}%")

# Set the components to 90 (99% threshold)
n=90
pca = PCA(n_components=n) 

# Fit the model to our data and extract the results
final_pca = pca.fit_transform(df_pca)

# Create a dataframe from the final pca dataset
X_train_pca = pd.DataFrame(data = final_pca, columns = [f'Component {i}' for i in range(1,n+1)])
X_train_pca

# Save target and features for training to drive
X_train_pca.to_csv('/content/drive/MyDrive/dataset/variables/pca/x_train_pca.csv', index=False,header =None)

"""### Test Dataset"""

# Apply to test dataset
X_test_5_pca = pca.transform(X_test_5)

X_test_5_pca = pd.DataFrame(data = X_test_5_pca, columns = [f'Component {i}' for i in range(1,n+1)])
X_test_5_pca

X_test_7_pca = pca.transform(X_test_7)
X_test_7_pca = pd.DataFrame(data = X_test_7_pca, columns = [f'Component {i}' for i in range(1,n+1)])
##
X_test_8_pca = pca.transform(X_test_8)
X_test_8_pca = pd.DataFrame(data = X_test_8_pca, columns = [f'Component {i}' for i in range(1,n+1)])
##
X_test_9_pca = pca.transform(X_test_9)
X_test_9_pca = pd.DataFrame(data = X_test_9_pca, columns = [f'Component {i}' for i in range(1,n+1)])
##
X_test_10_pca = pca.transform(X_test_10)
X_test_10_pca = pd.DataFrame(data = X_test_10_pca, columns = [f'Component {i}' for i in range(1,n+1)])

##
X_test_5_pca.to_csv('/content/drive/MyDrive/dataset/variables/pca/x_test_5_pca.csv', index=False, header =None)
X_test_7_pca.to_csv('/content/drive/MyDrive/dataset/variables/pca/x_test_7_pca.csv', index=False,header =None)
X_test_8_pca.to_csv('/content/drive/MyDrive/dataset/variables/pca/x_test_8_pca.csv', index=False,header =None)
X_test_9_pca.to_csv('/content/drive/MyDrive/dataset/variables/pca/x_test_9_pca.csv', index=False,header =None)
X_test_10_pca.to_csv('/content/drive/MyDrive/dataset/variables/pca/x_test_10_pca.csv', index=False,header =None)

"""# Training"""

# Read original variables from drive
X_train = pd.read_csv('/content/drive/MyDrive/dataset/variables/original/x_train.csv',header =None)
y_train = pd.read_csv('/content/drive/MyDrive/dataset/variables/original/y_train.csv',header =None)

# Read pca variables from drive
X_train_pca = pd.read_csv('/content/drive/MyDrive/dataset/variables/pca/x_train_pca.csv',header =None)

"""## Artificial Neural Network (ANN)

---


"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Sequential, layers
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from tqdm import tqdm
from tensorflow.keras import layers
from keras.layers import Dense, Dropout, Activation

import random
import numpy as np
import tensorflow as tf
random.seed(1234)
np.random.seed(1234)
tf.random.set_seed(1234)

# Importing variables
X_train = pd.read_csv('/content/drive/MyDrive/dataset/variables/original/x_train.csv',header =None)
X_train_pca = pd.read_csv('/content/drive/MyDrive/dataset/variables/pca/x_train_pca.csv',header =None)
y_train = pd.read_csv('/content/drive/MyDrive/dataset/variables/original/y_train.csv',header =None)

X_train = X_train.to_numpy()
X_train_pca = X_train_pca.to_numpy()
y_train = y_train.to_numpy()

print(X_train.shape)
print(X_train_pca.shape)
print(y_train.shape)

initializer = tf.keras.initializers.TruncatedNormal(mean=0.0, stddev=0.05, seed=1234)

"""### Single Layer Perceptron

#### Original
"""

# SINGLE-LAYER PERCEPTRON (SLP) NEURAL NETWORK
'''
    A Feed-forward neural network with:
    - 114 inputs
    - 1 hidden layer (number of nodes: 2/3 of input + output) 
    - 10 outputs

    '''
# Define model
slp_1 = Sequential()
slp_1.add(layers.Dense(80, activation='relu',kernel_initializer=initializer, input_shape=(114,)))
slp_1.add(layers.Dense(10, activation='softmax'))

# Compile the model
adam = tf.keras.optimizers.Adam(learning_rate=0.001, beta_1 = 0.9, beta_2 = 0.999, epsilon=1e-08)
slp_1.compile(optimizer=adam, loss=tf.keras.losses.SparseCategoricalCrossentropy(),
              metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])

slp_1.summary()

# Fit the model
history_slp_1 = slp_1.fit(X_train, y_train, epochs=300, batch_size=128)
loss_sln_1, acc_sln_1 = slp_1.evaluate(X_train, y_train)

print("Train Loss: %f, Train Accuracy: %f" % (loss_sln_1, acc_sln_1))

#Single Layer Perceptron Accuracy Plot 
import matplotlib.pyplot as plt

plt.plot(history_slp_1.history['sparse_categorical_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train'], loc = 'upper left')
plt.show()

#Single Layer Perceptron Loss Plot
plt.plot(history_slp_1.history['loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train'], loc = 'upper left')
plt.show()

history_slp_1.history.keys()

import numpy as np

y_pred = np.argmax(slp_1.predict(X_train), axis=-1)
y_pred

#Confusion Matrix SLP_1
cm_slp_1 = confusion_matrix(y_train, y_pred)

# Plot confusion matrix in a beautiful manner
fig = plt.figure(figsize=(8,6))
ax= plt.subplot()
sns.heatmap(cm_slp_1, annot=True, ax = ax, cmap ='YlOrBr',  fmt = 'g'); #annot=True to annotate cells
# labels, title and ticks
ax.set_xlabel('Predicted', fontsize=20)
ax.xaxis.set_label_position('bottom')
plt.xticks(rotation=90)
ax.xaxis.set_ticklabels(['0', '1', '2', '3', '4', '5', '6','7', '8', '9'], fontsize = 10)
ax.xaxis.tick_bottom()

ax.set_ylabel('True', fontsize=20)
ax.yaxis.set_ticklabels(['0', '1', '2', '3', '4', '5', '6','7', '8', '9'], fontsize = 10)
plt.yticks(rotation=0)

plt.title('Confusion Matrix of Single Layer Perceptron on PCA dataset', fontsize=20)

plt.savefig('ConMat24.png')
plt.show()

"""#### PCA"""

# SINGLE-LAYER PERCEPTRON (SLP) NEURAL NETWORK
'''
    A Feed-forward neural network with:
    - 90 inputs
    - 1 hidden layer (number of nodes: 2/3 of input + output) 
    - 10 outputs

    '''
# Define model
sln_1_pca = Sequential()
sln_1_pca.add(layers.Dense(70, activation = 'relu',kernel_initializer=initializer, input_shape=(90,)))
sln_1_pca.add(layers.Dense(10, activation='softmax'))

# Compile the model
adam = tf.keras.optimizers.Adam(learning_rate=0.001, beta_1 = 0.9, beta_2 = 0.999, epsilon=1e-08)
sln_1_pca.compile(optimizer=adam, loss=tf.keras.losses.SparseCategoricalCrossentropy(),
              metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])

# Fit the model
history_sln_1_pca = sln_1_pca.fit(X_train_pca, y_train, epochs=200, batch_size=128)
loss_sln_1_pca, acc_sln_1_pca = sln_1_pca.evaluate(X_train_pca, y_train)

print("Train Loss: %f, Train Accuracy: %f" % (loss_sln_1_pca, acc_sln_1_pca))

plt.plot(history_sln_1_pca.history['sparse_categorical_accuracy'])
plt.title('Single Layer Perceptron Neural Network Accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train'], loc = 'upper left')
plt.show()

#Single Layer Perceptron Loss Plot
plt.plot(history_sln_1_pca.history['loss'])
plt.title('Single Layer Perceptron Neural Network Loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train'], loc = 'upper left')
plt.show()

y_pred_sln_1_pca = np.argmax(sln_1_pca.predict(X_train_pca), axis=-1)
y_pred_sln_1_pca

#Confusion Matrix
cm_sln_1_pca = confusion_matrix(y_train, y_pred_sln_1_pca)

# Plot confusion matrix in a beautiful manner
fig = plt.figure(figsize=(8,6))
ax= plt.subplot()
sns.heatmap(cm_sln_1_pca, annot=True, ax = ax, cmap ='YlOrBr',  fmt = 'g'); #annot=True to annotate cells
# labels, title and ticks
ax.set_xlabel('Predicted', fontsize=20)
ax.xaxis.set_label_position('bottom')
plt.xticks(rotation=90)
ax.xaxis.set_ticklabels(['0', '1', '2', '3', '4', '5', '6','7', '8', '9'], fontsize = 10)
ax.xaxis.tick_bottom()

ax.set_ylabel('True', fontsize=20)
ax.yaxis.set_ticklabels(['0', '1', '2', '3', '4', '5', '6','7', '8', '9'], fontsize = 10)
plt.yticks(rotation=0)

plt.title('Confusion Matrix of Single Layer Perceptron on PCA dataset', fontsize=20)

plt.savefig('ConMat24.png')
plt.show()

"""### Multi-layer Perceptron

#### Original
"""

# MULTI-LAYER PERCEPTRON (MLP) NEURAL NETWORK
'''
    A Feed-forward neural network with:
    - 90 inputs
    - 2 hidden layer (max number of nodes: 2/3 of input + output) 
    - 10 outputs

    '''

mlp_1 = Sequential()
mlp_1.add(layers.Dense(50, activation='relu',kernel_initializer= initializer, input_shape=(114,)))
mlp_1.add(layers.Dense(30, activation='relu'))
mlp_1.add(layers.Dense(10, activation='softmax'))

# Compile the model
adam = tf.keras.optimizers.Adam(learning_rate=0.001, beta_1 = 0.9, beta_2 = 0.999, epsilon=1e-08)
mlp_1.compile(optimizer=adam, loss=tf.keras.losses.SparseCategoricalCrossentropy(),
              metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])

# Fit the model
history_mlp_1 = mlp_1.fit(X_train, y_train, epochs=300, batch_size=128)
loss_mlp_1, acc_mlp_1 = mlp_1.evaluate(X_train, y_train)

print("Train Loss: %f, Train Accuracy: %f" % (loss_mlp_1, acc_mlp_1))

##Accuracy Plot -Multi Layer Perceptron 
plt.plot(history_mlp_1.history['sparse_categorical_accuracy'])
plt.title('Multi Layer Perceptron Accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train'], loc = 'upper left')
plt.show()

#Multi Layer Perceptron Loss Plot
plt.plot(history_mlp_1.history['loss'])
plt.title('Multi Layer Perceptron Loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train'], loc = 'upper left')
plt.show()

y_pred_mlp_1 = np.argmax(mlp_1.predict(X_train), axis=-1)
y_pred_mlp_1

#Confusion Matrix
cm_mlp_1 = confusion_matrix(y_train, y_pred_mlp_1)

# Plot confusion matrix in a beautiful manner
fig = plt.figure(figsize=(8,6))
ax= plt.subplot()
sns.heatmap(cm_mlp_1, annot=True, ax = ax, cmap ='YlOrBr', fmt = 'g'); #annot=True to annotate cells
# labels, title and ticks
ax.set_xlabel('Predicted', fontsize=20)
ax.xaxis.set_label_position('bottom')
plt.xticks(rotation=90)
ax.xaxis.set_ticklabels(['0', '1', '2', '3', '4', '5', '6','7', '8', '9'], fontsize = 10)
ax.xaxis.tick_bottom()

ax.set_ylabel('True', fontsize=20)
ax.yaxis.set_ticklabels(['0', '1', '2', '3', '4', '5', '6','7', '8', '9'], fontsize = 10)
plt.yticks(rotation=0)

plt.title('Confusion Matrix of Multi Layer Perceptron', fontsize=20)

plt.show()

"""#### PCA"""

##MLP with PCA selected features
initializer = tf.keras.initializers.TruncatedNormal(mean=0.0, stddev=0.05, seed=1234)

mlp_1_pca = Sequential()
mlp_1_pca.add(layers.Dense(40, activation='relu',kernel_initializer= initializer, input_shape=(90,)))
mlp_1_pca.add(layers.Dense(30, activation='relu'))
mlp_1_pca.add(layers.Dense(10, activation='softmax'))

# Compile the model
adam = tf.keras.optimizers.Adam(learning_rate=0.001, beta_1 = 0.9, beta_2 = 0.999, epsilon=1e-08)
mlp_1_pca.compile(optimizer=adam, loss=tf.keras.losses.SparseCategoricalCrossentropy(),
              metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])

# Fit the model
history_mlp_1_pca = mlp_1_pca.fit(X_train_pca, y_train, epochs=200, batch_size=128)
loss_mlp_1_pca, acc_mlp_1_pca = mlp_1_pca.evaluate(X_train_pca, y_train)

print("Train Loss: %f, Train Accuracy: %f" % (loss_mlp_1_pca, acc_mlp_1_pca))

#Multi Layer Perceptron PCA Accuracy Plot
plt.plot(history_mlp_1_pca.history['sparse_categorical_accuracy'])
plt.title('Multi Layer Perceptron (PCAselected features) Accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train'], loc = 'upper left')
plt.show()

#Multi Layer Perceptron Loss Plot
plt.plot(history_mlp_1_pca.history['loss'])
plt.title('Multi Layer Perceptron Loss(PCA selected features)')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train'], loc = 'upper left')
plt.show()

#Label Prediction
y_pred_mlp_1_pca = np.argmax(mlp_1_pca.predict(X_train_pca), axis=-1)
y_pred_mlp_1_pca

#Confusion Matrix
cm_mlp_1_pca = confusion_matrix(y_train, y_pred_mlp_1_pca)

# Plot confusion matrix in a beautiful manner
fig = plt.figure(figsize=(8,6))
ax= plt.subplot()
sns.heatmap(cm_mlp_1_pca, annot=True, ax = ax, cmap ='YlOrBr' , fmt = 'g'); #annot=True to annotate cells
# labels, title and ticks
ax.set_xlabel('Predicted', fontsize=20)
ax.xaxis.set_label_position('bottom')
plt.xticks(rotation=90)
ax.xaxis.set_ticklabels(['0', '1', '2', '3', '4', '5', '6','7', '8', '9'], fontsize = 10)
ax.xaxis.tick_bottom()

ax.set_ylabel('True', fontsize=20)
ax.yaxis.set_ticklabels(['0', '1', '2', '3', '4', '5', '6','7', '8', '9'], fontsize = 10)
plt.yticks(rotation=0)

plt.title('Confusion Matrix Multi Layer Perceptron on PCA dataset', fontsize=20)

plt.savefig('ConMat24.png')
plt.show()

"""### MLP w/Dropout

#### Original
"""

mlp_2 = Sequential()
mlp_2.add(layers.Dense(40, activation='relu',kernel_initializer= initializer, input_shape=(114,)))
mlp_2.add(Dropout(0.5))
mlp_2.add(layers.Dense(30, activation='relu'))
mlp_2.add(Dropout(0.5))
mlp_2.add(layers.Dense(10, activation='softmax'))

# Compile the model
adam = tf.keras.optimizers.Adam(learning_rate=0.001, beta_1 = 0.9, beta_2 = 0.999, epsilon=1e-08)
mlp_2.compile(optimizer=adam, loss=tf.keras.losses.SparseCategoricalCrossentropy(),
              metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])

# Fit the model
history_mlp_2 = mlp_2.fit(X_train, y_train, epochs=300, batch_size=128)
loss_mlp_2, acc_mlp_2 = mlp_2.evaluate(X_train, y_train)

print("Train Loss: %f, Train Accuracy: %f" % (loss_mlp_2, acc_mlp_2))

#Multi Layer Perceptron with Dropout Accuracy Plot
plt.plot(history_mlp_2.history['sparse_categorical_accuracy'])
plt.title('Multi Layer Perceptron with Dropout Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('epoch')
plt.legend(['train'], loc = 'upper left')
plt.show()

#Multi Layer Perceptron with Dropout Loss Plot 
plt.plot(history_mlp_2.history['loss'])
plt.title('Multi Layer Perceptron with Dropout Loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train'], loc = 'upper left')
plt.show()

y_pred_mlp_2 = np.argmax(mlp_2.predict(X_train), axis=-1)
y_pred_mlp_2

#Confusion Matrix
cm_mlp_2 = confusion_matrix(y_train, y_pred_mlp_2)

# Plot confusion matrix in a beautiful manner
fig = plt.figure(figsize=(8,6))
ax= plt.subplot()
sns.heatmap(cm_mlp_2, annot=True, ax = ax,cmap ='YlOrBr', fmt = 'g'); #annot=True to annotate cells
# labels, title and ticks
ax.set_xlabel('Predicted', fontsize=20)
ax.xaxis.set_label_position('bottom')
plt.xticks(rotation=90)
ax.xaxis.set_ticklabels(['0', '1', '2', '3', '4', '5', '6','7', '8', '9'], fontsize = 10)
ax.xaxis.tick_bottom()

ax.set_ylabel('True', fontsize=20)
ax.yaxis.set_ticklabels(['0', '1', '2', '3', '4', '5', '6','7', '8', '9'], fontsize = 10)
plt.yticks(rotation=0)

plt.title('Confusion Matrix of Multi Layer Perceptron with Dropout', fontsize=20)
plt.show()

"""####PCA"""

mlp_2_pca = Sequential()
mlp_2_pca.add(layers.Dense(40, activation='relu',kernel_initializer= initializer, input_shape=(90,)))
mlp_2_pca.add(Dropout(0.5))
mlp_2_pca.add(layers.Dense(30, activation='relu'))
mlp_2_pca.add(Dropout(0.5))
mlp_2_pca.add(layers.Dense(10, activation='softmax'))

# Compile the model
adam = tf.keras.optimizers.Adam(learning_rate=0.001, beta_1 = 0.9, beta_2 = 0.999, epsilon=1e-08)
mlp_2_pca.compile(optimizer=adam, loss=tf.keras.losses.SparseCategoricalCrossentropy(),
              metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])

# Fit the model
history_mlp_2_pca = mlp_2_pca.fit(X_train_pca, y_train, epochs=200, batch_size=128)
loss_mlp_2_pca, acc_mlp_2_pca = mlp_2_pca.evaluate(X_train_pca, y_train)

print("Train Loss: %f, Train Accuracy: %f" % (loss_mlp_2_pca, acc_mlp_2_pca))

#Multi Layer Perceptron with Dropout on PCA dataset Accuracy Plot
plt.plot(history_mlp_2_pca.history['sparse_categorical_accuracy'])
plt.title('Multi Layer Perceptron with Dropout on PCA dataset Acuracy Plot')
plt.ylabel('Accuracy')
plt.xlabel('epoch')
plt.legend(['train'], loc = 'upper left')
plt.show()

##Multi Layer Perceptron with Dropout on PCA dataset Loss Plot
plt.plot(history_mlp_2_pca.history['loss'])
plt.title('Multi Layer Perceptron with Dropout on PCA Dataset Loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train'], loc = 'upper left')
plt.show()

y_pred_mlp_2_pca = np.argmax(mlp_2_pca.predict(X_train_pca), axis=-1)
y_pred_mlp_2_pca

#Confusion Matrix
cm_mlp_2_pca = confusion_matrix(y_train, y_pred_mlp_2_pca)

# Plot confusion matrix in a beautiful manner
fig = plt.figure(figsize=(8,6))
ax= plt.subplot()
sns.heatmap(cm_mlp_2_pca, annot=True, ax = ax, cmap ='YlOrBr', fmt = 'g'); #annot=True to annotate cells
# labels, title and ticks
ax.set_xlabel('Predicted', fontsize=20)
ax.xaxis.set_label_position('bottom')
plt.xticks(rotation=90)
ax.xaxis.set_ticklabels(['0', '1', '2', '3', '4', '5', '6','7', '8', '9'], fontsize = 10)
ax.xaxis.tick_bottom()

ax.set_ylabel('True', fontsize=20)
ax.yaxis.set_ticklabels(['0', '1', '2', '3', '4', '5', '6','7', '8', '9'], fontsize = 10)
plt.yticks(rotation=0)

plt.title('Confusion Matrix of Multi Layer Perceptron with Dropout on PCA dataset', fontsize=20)

plt.savefig('ConMat24.png')
plt.show()

"""### MLP w/Dropout and weights

#### Orginal
"""

!pip3 install ann_visualizer
!pip install graphviz

from sklearn.utils import class_weight

y_train = y_train.flatten()

class_weights = class_weight.compute_class_weight(
                                        class_weight = "balanced",
                                        classes = np.unique(y_train),
                                        y = y_train                                                    
                                    )
class_weights = dict(zip(np.unique(y_train), class_weights))
class_weights

mlp_3 = Sequential()
mlp_3.add(layers.Dense(50, activation='relu', kernel_initializer= initializer, input_shape=(114,)))
mlp_3.add(Dropout(0.7))
mlp_3.add(layers.Dense(30, activation='relu'))
mlp_3.add(Dropout(0.7))
mlp_3.add(layers.Dense(10, activation='softmax'))

# Compile the model
adam = tf.keras.optimizers.Adam(learning_rate=0.001, beta_1 = 0.9, beta_2 = 0.999, epsilon=1e-08)
mlp_3.compile(optimizer=adam, loss=tf.keras.losses.SparseCategoricalCrossentropy(),
              metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])

# Fit the model
history_mlp_3 = mlp_3.fit(X_train, y_train, epochs=400, batch_size=64, class_weight=class_weights)
loss_mlp_3, acc_mlp_3 = mlp_3.evaluate(X_train, y_train)

print("Train Loss: %f, Train Accuracy: %f" % (loss_mlp_3, acc_mlp_3))

#Multi Layer Perceptron with Dropout and weights Accuracy Plot 
plt.plot(history_mlp_3.history['sparse_categorical_accuracy'])
plt.title('Multi Layer Perceptron with Dropout and weights Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('epoch')
plt.legend(['train'], loc = 'upper left')
plt.show()

##Multi Layer Perceptron with Dropout and weights Loss Plot
plt.plot(history_mlp_3.history['loss'])
plt.title('Multi Layer Perceptron with Dropout and weight Loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train'], loc = 'upper left')
plt.show()

y_test.dtype

from ann_visualizer.visualize import ann_viz;

ann_viz(mlp_3, title="Final Model")

#Label Predictions
y_pred_mlp_3 = np.argmax(mlp_3.predict(X_train), axis=-1)
y_pred_mlp_3

#Confusion Matrix 
cm_mlp_3 = confusion_matrix(y_train, y_pred_mlp_3)

# Plot confusion matrix in a beautiful manner
fig = plt.figure(figsize=(8,6))
ax= plt.subplot()
sns.heatmap(cm_mlp_3, annot=True, ax = ax,cmap ='YlOrBr', fmt = 'g'); #annot=True to annotate cells
# labels, title and ticks
ax.set_xlabel('Predicted', fontsize=20)
ax.xaxis.set_label_position('bottom')
plt.xticks(rotation=90)
ax.xaxis.set_ticklabels(['0', '1', '2', '3', '4', '5', '6','7', '8', '9'], fontsize = 10)
ax.xaxis.tick_bottom()

ax.set_ylabel('True', fontsize=20)
ax.yaxis.set_ticklabels(['0', '1', '2', '3', '4', '5', '6','7', '8', '9'], fontsize = 10)
plt.yticks(rotation=0)

plt.title('Confusion Matrix of Multi Layer Perceptron with Dropout and weights', fontsize=20)

plt.show()

"""####PCA"""

mlp_3_pca = Sequential()
mlp_3_pca.add(layers.Dense(40, activation='relu',kernel_initializer= initializer, input_shape=(90,)))
mlp_3_pca.add(Dropout(0.7))
mlp_3_pca.add(layers.Dense(30, activation='relu'))
mlp_3_pca.add(Dropout(0.7))
mlp_3_pca.add(layers.Dense(10, activation='softmax'))

# Compile the model
adam = tf.keras.optimizers.Adam(learning_rate=0.01, beta_1 = 0.9, beta_2 = 0.999, epsilon=1e-08)
mlp_3_pca.compile(optimizer=adam, loss=tf.keras.losses.SparseCategoricalCrossentropy(),
              metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])

# Fit the model
history_mlp_3_pca = mlp_3_pca.fit(X_train_pca, y_train, epochs=200, batch_size=64, class_weight=class_weights)
loss_mlp_3_pca, acc_mlp_3_pca = mlp_3_pca.evaluate(X_train_pca, y_train)

print("Train Loss: %f, Train Accuracy: %f" % (loss_mlp_3_pca, acc_mlp_3_pca))

##Multi Layer Perceptron with Dropout and Weights on PCA dataset Accuracy Plot
plt.plot(history_mlp_3_pca.history['sparse_categorical_accuracy'])
plt.title('Multi Layer Perceptron with Dropout and weights on PCA Dataset Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('epoch')
plt.legend(['train'], loc = 'upper left')
plt.show()

##Multi Layer Perceptron with Dropout and Weights on PCA dataset Loss Plot
plt.plot(history_mlp_3_pca.history['loss'])
plt.title('Multi Layer Perceptron with Dropout on PCA Dataset Loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train'], loc = 'upper left')
plt.show()

#Label Predictions
y_pred_mlp_3_pca = np.argmax(mlp_3_pca.predict(X_train_pca), axis=-1)
y_pred_mlp_3_pca

##Confusion Matrix
cm_mlp_3_pca = confusion_matrix(y_train, y_pred_mlp_3_pca)

# Plot confusion matrix in a beautiful manner
fig = plt.figure(figsize=(8,6))
ax= plt.subplot()
sns.heatmap(cm_mlp_3_pca, annot=True, ax = ax,cmap ='YlOrBr', fmt = 'g'); #annot=True to annotate cells
# labels, title and ticks
ax.set_xlabel('Predicted', fontsize=20)
ax.xaxis.set_label_position('bottom')
plt.xticks(rotation=90)
ax.xaxis.set_ticklabels(['0', '1', '2', '3', '4', '5', '6','7', '8', '9'], fontsize = 10)
ax.xaxis.tick_bottom()

ax.set_ylabel('True', fontsize=20)
ax.yaxis.set_ticklabels(['0', '1', '2', '3', '4', '5', '6','7', '8', '9'], fontsize = 10)
plt.yticks(rotation=0)

plt.title('Confusion Matrix of Multi Layer Perceptron with Dropout and weights on PCA dataset', fontsize=20)

plt.show()

"""## Convolutional Neural Network (CNN)

### Image Pre-processing

#### Train
"""

#Unzip the image folder
!unzip '/content/drive/MyDrive/dataset/png.zip' -d '/'

# Define new audio path
png_path = os.path.join(us8k_path, '/content/UrbanSound8K/png')

import cv2

img_array, labels, height_array, width_array = [], [], [], []

# Loop over all IDs and read each image in one by one
folds = 'fold1 fold2 fold3 fold4 fold6'.split()
for f in folds:
    for filename in os.listdir(png_path + str('/') + str(f)):
        imagename = f'{png_path}/{f}/{filename}'
        image = cv2.imread (imagename)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        dim = (30, 30)
        image = cv2.resize(image, dim) # resize image
        img_array.append (image)
        with open("/content/UrbanSound8K/metadata/UrbanSound8K.csv", "r") as csvfile:
            reader = csv.reader(csvfile)
            next(reader)
            for row in reader:
              if filename == f'{str(row[0])[:-4]}.png':
                labels.append(int(row[6]))

X_train = np.array(img_array)
y_train = np.array(labels)

# Saving NumPy array as a file
np.save('/content/drive/MyDrive/dataset/variables/cnn/X_train', X_train)

# Saving NumPy array as a file
np.save('/content/drive/MyDrive/dataset/variables/cnn/y_train', y_train)

# Read cnn variables from drive
X_train_img = np.load('/content/drive/MyDrive/dataset/variables/cnn/X_train.npy')
y_train_img = np.load('/content/drive/MyDrive/dataset/variables/cnn/y_train.npy')

from google.colab.patches import cv2_imshow
cv2_imshow(X_train_img[0])

print(X_train_img.shape)
print(y_train_img.shape)

X_train_img = X_train_img.reshape(X_train_img.shape[0], X_train_img.shape[1], X_train_img.shape[2], 1)
print(X_train_img.shape)

# confirm pixel range is 0-255
print('Data Type: %s' % X_train_img.dtype)
print('Min: %.3f, Max: %.3f' % (X_train_img.min(), X_train_img.max()))

# convert from integers to floats
X_train_img = X_train_img.astype('float32')

from keras.preprocessing.image import ImageDataGenerator

# creating the image data generator [1.0/255.0 = 0.00392156862]
datagen = ImageDataGenerator(featurewise_center=True,
                             featurewise_std_normalization=True)
datagen.fit(X_train_img)

# preparing an iterator for scaling images
train_iterator = datagen.flow(X_train_img, y_train_img, batch_size=64)
print('Batches train=%d' % len(train_iterator))

# confirming- the scaling works
# demonstrate effect on a single batch of samples
# get a batch
batchX, batchy = train_iterator.next()
# pixel stats in the batch
print(batchX.shape, batchX.mean(), batchX.std())
# demonstrate effect on entire training dataset
iterator = datagen.flow(X_train_img, y_train_img, batch_size=len(X_train_img), shuffle=False)
# get a batch
batchX, batchy = iterator.next()
# pixel stats in the batch
print(batchX.shape, batchX.mean(), batchX.std())

"""#### Test"""

img_array2, labels2, height_array2, width_array2 = [], [], [], []

# Loop over all IDs and read each image in one by one
folds = 'fold5'.split()
for f in folds:
  for filename in os.listdir(png_path + str('/') + str(f)):
      imagename = f'{png_path}/{f}/{filename}'
      image = cv2.imread (imagename)
      image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
      dim = (30, 30)
      image = cv2.resize(image, dim) # resize image
      img_array2.append (image)
      with open("/content/UrbanSound8K/metadata/UrbanSound8K.csv", "r") as csvfile:
          reader = csv.reader(csvfile)
          next(reader)
          for row in reader:
            if filename == f'{str(row[0])[:-4]}.png':
              labels2.append(int(row[6]))

X_test_5 = np.array(img_array2)
y_test_5 = np.array(labels2)

img_array3, labels3, height_array3, width_array3 = [], [], [], []

# Loop over all IDs and read each image in one by one
folds = 'fold7'.split()
for f in folds:
  for filename in os.listdir(png_path + str('/') + str(f)):
      imagename = f'{png_path}/{f}/{filename}'
      image = cv2.imread (imagename)
      image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
      dim = (30, 30)
      image = cv2.resize(image, dim) # resize image
      img_array3.append (image)
      with open("/content/UrbanSound8K/metadata/UrbanSound8K.csv", "r") as csvfile:
          reader = csv.reader(csvfile)
          next(reader)
          for row in reader:
            if filename == f'{str(row[0])[:-4]}.png':
              labels3.append(int(row[6]))

X_test_7 = np.array(img_array3)
y_test_7 = np.array(labels3)

img_array4, labels4, height_array4, width_array4 = [], [], [], []

# Loop over all IDs and read each image in one by one
folds = 'fold8'.split()
for f in folds:
  for filename in os.listdir(png_path + str('/') + str(f)):
      imagename = f'{png_path}/{f}/{filename}'
      image = cv2.imread (imagename)
      image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
      dim = (30, 30)
      image = cv2.resize(image, dim) # resize image
      img_array4.append (image)
      with open("/content/UrbanSound8K/metadata/UrbanSound8K.csv", "r") as csvfile:
          reader = csv.reader(csvfile)
          next(reader)
          for row in reader:
            if filename == f'{str(row[0])[:-4]}.png':
              labels4.append(int(row[6]))

X_test_8 = np.array(img_array4)
y_test_8 = np.array(labels4)

img_array5, labels5, height_array5, width_array5 = [], [], [], []

# Loop over all IDs and read each image in one by one
folds = 'fold9'.split()
for f in folds:
  for filename in os.listdir(png_path + str('/') + str(f)):
      imagename = f'{png_path}/{f}/{filename}'
      image = cv2.imread (imagename)
      image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
      dim = (30, 30)
      image = cv2.resize(image, dim) # resize image
      img_array5.append (image)
      with open("/content/UrbanSound8K/metadata/UrbanSound8K.csv", "r") as csvfile:
          reader = csv.reader(csvfile)
          next(reader)
          for row in reader:
            if filename == f'{str(row[0])[:-4]}.png':
              labels5.append(int(row[6]))

X_test_9 = np.array(img_array5)
y_test_9 = np.array(labels5)

img_array6, labels6, height_array6, width_array6 = [], [], [], []

# Loop over all IDs and read each image in one by one
folds = 'fold10'.split()
for f in folds:
  for filename in os.listdir(png_path + str('/') + str(f)):
      imagename = f'{png_path}/{f}/{filename}'
      image = cv2.imread (imagename)
      image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
      dim = (30, 30)
      image = cv2.resize(image, dim) # resize image
      img_array6.append (image)
      with open("/content/UrbanSound8K/metadata/UrbanSound8K.csv", "r") as csvfile:
          reader = csv.reader(csvfile)
          next(reader)
          for row in reader:
            if filename == f'{str(row[0])[:-4]}.png':
              labels6.append(int(row[6]))

X_test_10 = np.array(img_array6)
y_test_10 = np.array(labels6)

# Saving NumPy array as a file
np.save('/content/drive/MyDrive/dataset/variables/cnn/X_test_5', X_test_5)
np.save('/content/drive/MyDrive/dataset/variables/cnn/X_test_7', X_test_7)
np.save('/content/drive/MyDrive/dataset/variables/cnn/X_test_8', X_test_8)
np.save('/content/drive/MyDrive/dataset/variables/cnn/X_test_9', X_test_9)
np.save('/content/drive/MyDrive/dataset/variables/cnn/X_test_10', X_test_10)

# Saving NumPy array as a file
np.save('/content/drive/MyDrive/dataset/variables/cnn/y_test_5', y_test_5)
np.save('/content/drive/MyDrive/dataset/variables/cnn/y_test_7', y_test_7)
np.save('/content/drive/MyDrive/dataset/variables/cnn/y_test_8', y_test_8)
np.save('/content/drive/MyDrive/dataset/variables/cnn/y_test_9', y_test_9)
np.save('/content/drive/MyDrive/dataset/variables/cnn/y_test_10', y_test_10)

# Read cnn variables from drive
X_test_img_5 = np.load('/content/drive/MyDrive/dataset/variables/cnn/X_test_5.npy')
y_test_img_5 = np.load('/content/drive/MyDrive/dataset/variables/cnn/y_test_5.npy')

##
X_test_img_7 = np.load('/content/drive/MyDrive/dataset/variables/cnn/X_test_7.npy')
y_test_img_7 = np.load('/content/drive/MyDrive/dataset/variables/cnn/y_test_7.npy')
##
X_test_img_8 = np.load('/content/drive/MyDrive/dataset/variables/cnn/X_test_8.npy')
y_test_img_8 = np.load('/content/drive/MyDrive/dataset/variables/cnn/y_test_8.npy')

##
X_test_img_9 = np.load('/content/drive/MyDrive/dataset/variables/cnn/X_test_9.npy')
y_test_img_9 = np.load('/content/drive/MyDrive/dataset/variables/cnn/y_test_9.npy')
##
X_test_img_10 = np.load('/content/drive/MyDrive/dataset/variables/cnn/X_test_10.npy')
y_test_img_10= np.load('/content/drive/MyDrive/dataset/variables/cnn/y_test_10.npy')

from google.colab.patches import cv2_imshow
cv2_imshow(X_test_img_5[0])

print(X_test_img_5.shape)
print(y_test_img_5.shape)
print(X_test_img_7.shape)
print(y_test_img_7.shape)
print(X_test_img_8.shape)
print(y_test_img_8.shape)
print(X_test_img_9.shape)
print(y_test_img_9.shape)
print(X_test_img_10.shape)
print(y_test_img_10.shape)

X_test_img_5 = X_test_img_5.reshape(X_test_img_5.shape[0], X_test_img_5.shape[1], X_test_img_5.shape[2], 1)
X_test_img_7 = X_test_img_7.reshape(X_test_img_7.shape[0], X_test_img_7.shape[1], X_test_img_7.shape[2], 1)
X_test_img_8 = X_test_img_8.reshape(X_test_img_8.shape[0], X_test_img_8.shape[1], X_test_img_8.shape[2], 1)
X_test_img_9 = X_test_img_9.reshape(X_test_img_9.shape[0], X_test_img_9.shape[1], X_test_img_9.shape[2], 1)
X_test_img_10 = X_test_img_10.reshape(X_test_img_10.shape[0], X_test_img_10.shape[1], X_test_img_10.shape[2], 1)
print(X_test_img_5.shape)
print(X_test_img_7.shape)
print(X_test_img_8.shape)
print(X_test_img_9.shape)
print(X_test_img_10.shape)

# confirm pixel range is 0-255
print('Data Type: %s' % X_test_img_5.dtype)
print('Data Type: %s' % X_test_img_7.dtype)
print('Data Type: %s' % X_test_img_8.dtype)
print('Data Type: %s' % X_test_img_9.dtype)
print('Data Type: %s' % X_test_img_10.dtype)
print('Min: %.3f, Max: %.3f' % (X_test_img_5.min(), X_test_img_5.max()))
print('Min: %.3f, Max: %.3f' % (X_test_img_7.min(), X_test_img_7.max()))
print('Min: %.3f, Max: %.3f' % (X_test_img_8.min(), X_test_img_8.max()))
print('Min: %.3f, Max: %.3f' % (X_test_img_9.min(), X_test_img_9.max()))
print('Min: %.3f, Max: %.3f' % (X_test_img_10.min(), X_test_img_10.max()))

# convert from integers to floats
X_test_img_5 = X_test_img_5.astype('float32')
X_test_img_7 = X_test_img_7.astype('float32')
X_test_img_8 = X_test_img_8.astype('float32')
X_test_img_9 = X_test_img_9.astype('float32')
X_test_img_10 = X_test_img_10.astype('float32')

from keras.preprocessing.image import ImageDataGenerator

# creating the image data generator [1.0/255.0 = 0.00392156862]
datagen = ImageDataGenerator(featurewise_center=True,
                             featurewise_std_normalization=True)
datagen.fit(X_test_img_5)

# preparing an iterator for scaling images
test_iterator_5 = datagen.flow(X_test_img_5, y_test_img_5, batch_size=64)
print('Batches train=%d' % len(test_iterator_5))

# confirming- the scaling works
# demonstrate effect on a single batch of samples
# get a batch
batchX_5, batchy_5 = test_iterator_5.next()
# pixel stats in the batch
print(batchX_5.shape, batchX_5.mean(), batchX_5.std())
# demonstrate effect on entire training dataset
iterator_5 = datagen.flow(X_test_img_5, y_test_img_5, batch_size=len(X_test_img_5), shuffle=False)
# get a batch
batchX_5, batchy_5 = iterator_5.next()
# pixel stats in the batch
print(batchX_5.shape, batchX_5.mean(), batchX_5.std())

from keras.preprocessing.image import ImageDataGenerator

# creating the image data generator [1.0/255.0 = 0.00392156862]
datagen = ImageDataGenerator(featurewise_center=True,
                             featurewise_std_normalization=True)
datagen.fit(X_test_img_7)

# preparing an iterator for scaling images
test_iterator_7 = datagen.flow(X_test_img_7, y_test_img_7, batch_size=64)
print('Batches train=%d' % len(test_iterator_7))

# confirming- the scaling works
# demonstrate effect on a single batch of samples
# get a batch
batchX_7, batchy_7 = test_iterator_7.next()
# pixel stats in the batch
print(batchX_7.shape, batchX_7.mean(), batchX_7.std())
# demonstrate effect on entire training dataset
iterator_7 = datagen.flow(X_test_img_7, y_test_img_7, batch_size=len(X_test_img_7), shuffle=False)
# get a batch
batchX_7, batchy_7 = iterator_7.next()
# pixel stats in the batch
print(batchX_7.shape, batchX_7.mean(), batchX_7.std())

from keras.preprocessing.image import ImageDataGenerator

# creating the image data generator [1.0/255.0 = 0.00392156862]
datagen = ImageDataGenerator(featurewise_center=True,
                             featurewise_std_normalization=True)
datagen.fit(X_test_img_8)

# preparing an iterator for scaling images
test_iterator_8 = datagen.flow(X_test_img_8, y_test_img_8, batch_size=64)
print('Batches train=%d' % len(test_iterator_8))

# confirming- the scaling works
# demonstrate effect on a single batch of samples
# get a batch
batchX_8, batchy_8 = test_iterator_8.next()
# pixel stats in the batch
print(batchX_8.shape, batchX_8.mean(), batchX_8.std())
# demonstrate effect on entire training dataset
iterator_8 = datagen.flow(X_test_img_8, y_test_img_8, batch_size=len(X_test_img_8), shuffle=False)
# get a batch
batchX_8, batchy_8 = iterator_8.next()
# pixel stats in the batch
print(batchX_8.shape, batchX_8.mean(), batchX_8.std())

from keras.preprocessing.image import ImageDataGenerator

# creating the image data generator [1.0/255.0 = 0.00392156862]
datagen = ImageDataGenerator(featurewise_center=True,
                             featurewise_std_normalization=True)
datagen.fit(X_test_img_9)

# preparing an iterator for scaling images
test_iterator_9 = datagen.flow(X_test_img_9, y_test_img_9, batch_size=64)
print('Batches train=%d' % len(test_iterator_9))

# confirming- the scaling works
# demonstrate effect on a single batch of samples
# get a batch
batchX_9, batchy_9 = test_iterator_9.next()
# pixel stats in the batch
print(batchX_9.shape, batchX_9.mean(), batchX_9.std())
# demonstrate effect on entire training dataset
iterator_9 = datagen.flow(X_test_img_9, y_test_img_9, batch_size=len(X_test_img_9), shuffle=False)
# get a batch
batchX_9, batchy_9 = iterator_9.next()
# pixel stats in the batch
print(batchX_9.shape, batchX_9.mean(), batchX_9.std())

from keras.preprocessing.image import ImageDataGenerator

# creating the image data generator [1.0/255.0 = 0.00392156862]
datagen = ImageDataGenerator(featurewise_center=True,
                             featurewise_std_normalization=True)
datagen.fit(X_test_img_10)

# preparing an iterator for scaling images
test_iterator_10 = datagen.flow(X_test_img_10, y_test_img_10, batch_size=64)
print('Batches train=%d' % len(test_iterator_10))

# confirming- the scaling works
# demonstrate effect on a single batch of samples
# get a batch
batchX_10, batchy_10 = test_iterator_10.next()
# pixel stats in the batch
print(batchX_10.shape, batchX_10.mean(), batchX_10.std())
# demonstrate effect on entire training dataset
iterator_10 = datagen.flow(X_test_img_10, y_test_img_10, batch_size=len(X_test_img_10), shuffle=False)
# get a batch
batchX_10, batchy_10 = iterator_10.next()
# pixel stats in the batch
print(batchX_10.shape, batchX_10.mean(), batchX_10.std())

"""### Models"""

from tensorflow.keras.layers import Dense, Activation, Flatten, Dropout, BatchNormalization
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras import regularizers, optimizers

"""#### Conv-Pool-Conv-Pool"""

import random
import numpy as np
import tensorflow as tf
random.seed(1234)
np.random.seed(1234)
tf.random.set_seed(1234)

# Conv-Pool-Conv-Pool Model
cnn_1 = Sequential([
    Conv2D(8, kernel_size=(3, 3), activation='relu',padding='same',input_shape=(30,30,1)),
    MaxPooling2D(pool_size=(2, 2),strides=2),
    Conv2D(16, kernel_size=(5, 5), activation='relu'),
    MaxPooling2D(pool_size=(2, 2),strides=2),
    Flatten(),
    Dense(400, activation='relu'),
    Dense(200, activation='relu'),
    Dense(10, activation='softmax')
])

#Model Compile
adam = optimizers.Adam(learning_rate=0.001, beta_1 = 0.9, beta_2 = 0.999, epsilon=1e-08)
cnn_1.compile(optimizer=adam,loss=tf.keras.losses.SparseCategoricalCrossentropy(),metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])
cnn_1.summary()

history_cnn_1 = cnn_1.fit(X_train_img, y_train_img, epochs=20, batch_size=32)
loss_cnn_1, acc_cnn_1 = cnn_1.evaluate(X_train_img, y_train_img)

print("Train Loss: %f, Train Accuracy: %f" % (loss_cnn_1, acc_cnn_1))

# Conv-Pool-Conv-Pool Accuracy Plot
plt.plot(history_cnn_1.history['sparse_categorical_accuracy'])
plt.title('Conv Pool-Conv-Pool Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('epoch')
plt.legend(['train'], loc = 'upper left')
plt.show()

# Conv-Pool-Conv-pool Loss Plot
plt.plot(history_cnn_1.history['loss'])
plt.title(' Conv-Pool-Conv-Pool Loss')
plt.ylabel('Loss')
plt.xlabel('epoch')
plt.legend(['train'], loc = 'upper left')
plt.show()

import sklearn.metrics as metrics

y_pred_ohe = cnn_1.predict(X_train_img)  
y_pred_labels = np.argmax(y_pred_ohe, axis=1)  # only necessary if output has one-hot-encoding, shape=(n_samples)

confusion_matrix_cnn_1 = metrics.confusion_matrix(y_true=y_train_img, y_pred=y_pred_labels)

import seaborn as sns
fig = plt.figure(figsize=(8,6))
ax= plt.subplot()

ax = sns.heatmap(confusion_matrix_cnn_1/np.sum(confusion_matrix_cnn_1), annot=True, cmap='Blues')

ax.set_title('Seaborn Confusion Matrix with labels\n\n');
ax.set_xlabel('\nPredicted Values')
ax.set_ylabel('Actual Values ');

## Ticket labels - List must be in alphabetical order
ax.xaxis.set_ticklabels(['0','1','2', '3', '4', '5', '6', '7', '8', '9'])
ax.yaxis.set_ticklabels(['0','1','2', '3', '4', '5', '6', '7', '8', '9'])

## Display the visualization of the Confusion Matrix.
plt.show()

from sklearn.metrics import classification_report
print('\nClassification Report\n')
print(classification_report(y_train_img, y_pred_labels, target_names=['0','1','2', '3', '4', '5', '6', '7', '8', '9']))



"""#### ConvConv - Pool - ConvConv - Pool"""

# ConvConv - Pool - ConvConv - Pool Model 
cnn_2 = Sequential([
    Conv2D(16, kernel_size=(3, 3), activation='relu',padding='same',input_shape=(30,30,1)),
    Conv2D(16, kernel_size=(3, 3), activation='relu',padding='same'),
    MaxPooling2D(pool_size=(2, 2),strides=2),
    Conv2D(32, kernel_size=(3, 3), activation='relu'),
    Conv2D(32, kernel_size=(3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2),strides=2),
    Flatten(),
    Dense(800, activation='relu'),    
    Dense(500, activation='relu'),
    Dense(10, activation='softmax')
])

#Model Compile
adam = optimizers.Adam(learning_rate=0.001, beta_1 = 0.9, beta_2 = 0.999, epsilon=1e-08)
cnn_2.compile(optimizer=adam,loss=tf.keras.losses.SparseCategoricalCrossentropy(),metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])
cnn_2.summary()

history_cnn_2 = cnn_2.fit(X_train_img, y_train_img, epochs=20 ,batch_size=32)
loss_cnn_2, acc_cnn_2 = cnn_2.evaluate(X_train_img, y_train_img)

print("Train Loss: %f, Train Accuracy: %f" % (loss_cnn_2, acc_cnn_2))

##CNN2 ConvConv-Pool-convConv-Pool Accuracy Plot
plt.plot(history_cnn_2.history['sparse_categorical_accuracy'])
plt.title(' ConvConv-Pool-convConv-Pool AccAccuracy')
plt.ylabel('Accuracy')
plt.xlabel('epoch')
plt.legend(['train'], loc = 'upper left')
plt.show()

##ConvConv-Pool-convConv-Pool Loss Plot
plt.plot(history_cnn_2.history['loss'])
plt.title('ConvConv-Pool-convConv-Pool Loss')
plt.ylabel('Loss')
plt.xlabel('epoch')
plt.legend(['train'], loc = 'upper left')
plt.show()

import sklearn.metrics as metrics

y_pred_ohe = cnn_2.predict(X_train_img)  
y_pred_labels = np.argmax(y_pred_ohe, axis=1)  # only necessary if output has one-hot-encoding, shape=(n_samples)

confusion_matrix_cnn_2 = metrics.confusion_matrix(y_true=y_train_img, y_pred=y_pred_labels)

# Plot confusion matrix in a beautiful manner
fig = plt.figure(figsize=(8,6))
ax= plt.subplot()
sns.heatmap(confusion_matrix_cnn_2, annot=True, ax = ax,cmap ='YlOrBr', fmt = 'g'); #annot=True to annotate cells
# labels, title and ticks
ax.set_xlabel('Predicted', fontsize=20)
ax.xaxis.set_label_position('bottom')
plt.xticks(rotation=90)
ax.xaxis.set_ticklabels(['0', '1', '2', '3', '4', '5', '6','7', '8', '9'], fontsize = 10)
ax.xaxis.tick_bottom()

ax.set_ylabel('True', fontsize=20)
ax.yaxis.set_ticklabels(['0', '1', '2', '3', '4', '5', '6','7', '8', '9'], fontsize = 10)
plt.yticks(rotation=0)

plt.title('Confusion Matrix of Conv Conv-Pool-ConvConv-Pool Network', fontsize=20)

plt.show()

from sklearn.metrics import classification_report
print('\nClassification Report\n')
print(classification_report(y_train_img, y_pred_labels, target_names=['0','1','2', '3', '4', '5', '6', '7', '8', '9']))

"""#### ConvConv - Pool - ConvConv - Pool Model with dropout"""

# ConvConv - Pool - ConvConv - Pool Model with dropout
cnn_3 = Sequential()

# Conv 1
cnn_3.add(Conv2D(32, (5, 5), padding='same',
                 input_shape=(30,30,1)))
cnn_3.add(Activation('relu'))

# Conv 2
cnn_3.add(Conv2D(64, (3, 3)))
cnn_3.add(Activation('relu'))

# Pool
cnn_3.add(MaxPooling2D(pool_size=(2, 2)))
cnn_3.add(BatchNormalization()) # Regularization

# Conv 3
cnn_3.add(Conv2D(128, (3, 3), padding='same'))
cnn_3.add(Activation('relu'))

# Conv 4
cnn_3.add(Conv2D(256, (3, 3)))
cnn_3.add(Activation('relu'))

# Pool
cnn_3.add(MaxPooling2D(pool_size=(2, 2)))
cnn_3.add(BatchNormalization()) # Regularization

# Flatten
cnn_3.add(Flatten())

# Dense
cnn_3.add(layers.Dense(6400, activation = 'relu'))
cnn_3.add(Dropout(0.5)) # Regularization
cnn_3.add(layers.Dense(4000, activation = 'relu'))
cnn_3.add(Dropout(0.5)) # Regularization
cnn_3.add(Dense(10, activation='softmax'))

adam = optimizers.Adam(learning_rate=0.001, beta_1 = 0.9, beta_2 = 0.999, epsilon=1e-08)
cnn_3.compile(optimizer=adam,loss=tf.keras.losses.SparseCategoricalCrossentropy(),metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])
cnn_3.summary()

history_cnn_3 = cnn_3.fit(X_train_img, y_train_img, epochs=30 ,batch_size=32)
loss_cnn_3, acc_cnn_3 = cnn_3.evaluate(X_train_img, y_train_img)

print("Train Loss: %f, Train Accuracy: %f" % (loss_cnn_3, acc_cnn_3))

##CNN3 ConvConv-Pool-convConv-Pool with Dropout Accuracy Plot
plt.plot(history_cnn_3.history['sparse_categorical_accuracy'])
plt.title(' ConvConv-Pool-convConv-Pool with dropout Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('epoch')
plt.legend(['train'], loc = 'upper left')
plt.show()

##CNN3 ConvConv-Pool-convConv-Pool with Dropout Loss Plot
plt.plot(history_cnn_3.history['loss'])
plt.title(' ConvConv-Pool-convConv-Pool with Dropout Loss')
plt.ylabel('Loss')
plt.xlabel('epoch')
plt.legend(['train'], loc = 'upper left')
plt.show()

import sklearn.metrics as metrics

y_pred_ohe_cnn_3 = cnn_3.predict(X_train_img)  
y_pred_labels_cnn_3 = np.argmax(y_pred_ohe_cnn_3, axis=1)  # only necessary if output has one-hot-encoding, shape=(n_samples)

confusion_matrix_cnn_3 = metrics.confusion_matrix(y_true=y_train_img, y_pred=y_pred_labels_cnn_3)

#Print Confusion Matrix 
print(confusion_matrix_cnn_3)

# Plot confusion matrix in a beautiful manner
fig = plt.figure(figsize=(16, 14))
ax= plt.subplot()
sns.heatmap(confusion_matrix_cnn_3, annot=True, ax = ax, fmt = 'g'); #annot=True to annotate cells
# labels, title and ticks
ax.set_xlabel('Predicted', fontsize=20)
ax.xaxis.set_label_position('bottom')
plt.xticks(rotation=90)
ax.xaxis.set_ticklabels(['0', '1', '2', '3', '4', '5', '6','7', '8', '9'], fontsize = 10)
ax.xaxis.tick_bottom()

ax.set_ylabel('True', fontsize=20)
ax.yaxis.set_ticklabels(['0', '1', '2', '3', '4', '5', '6','7', '8', '9'], fontsize = 10)
plt.yticks(rotation=0)

plt.title('Confusion Matrix of Multi Layer Perceptron with Dropout and weights on test dataset', fontsize=20)

plt.show()

##Classification Report 
from sklearn.metrics import classification_report
print('\nClassification Report of CNN_3\n')
print(classification_report(y_train, y_pred_labels_cnn_3, target_names=['0','1','2', '3', '4', '5', '6', '7', '8', '9']))

pip install visualkeras

import visualkeras
visualkeras.layered_view(cnn_3)

"""# Testing

## ANN
"""

# Importing variables
X_test_5 = pd.read_csv('/content/drive/MyDrive/dataset/variables/original/x_test_5.csv')
X_test_7 = pd.read_csv('/content/drive/MyDrive/dataset/variables/original/x_test_7.csv')
X_test_8 = pd.read_csv('/content/drive/MyDrive/dataset/variables/original/x_test_8.csv')
X_test_9 = pd.read_csv('/content/drive/MyDrive/dataset/variables/original/x_test_9.csv')
X_test_10 = pd.read_csv('/content/drive/MyDrive/dataset/variables/original/x_test_10.csv')
X_test_5_pca = pd.read_csv('/content/drive/MyDrive/dataset/variables/pca/x_test_5_pca.csv')
X_test_7_pca = pd.read_csv('/content/drive/MyDrive/dataset/variables/pca/x_test_7_pca.csv')
X_test_8_pca = pd.read_csv('/content/drive/MyDrive/dataset/variables/pca/x_test_8_pca.csv')
X_test_9_pca = pd.read_csv('/content/drive/MyDrive/dataset/variables/pca/x_test_9_pca.csv')
X_test_10_pca = pd.read_csv('/content/drive/MyDrive/dataset/variables/pca/x_test_10_pca.csv')
y_test_5 = pd.read_csv('/content/drive/MyDrive/dataset/variables/original/y_test_5.csv')
y_test_7 = pd.read_csv('/content/drive/MyDrive/dataset/variables/original/y_test_7.csv')
y_test_8 = pd.read_csv('/content/drive/MyDrive/dataset/variables/original/y_test_8.csv')
y_test_9 = pd.read_csv('/content/drive/MyDrive/dataset/variables/original/y_test_9.csv')
y_test_10 = pd.read_csv('/content/drive/MyDrive/dataset/variables/original/y_test_10.csv')

# Check the type of imported Variables
type(X_test_5)

# Transform into array
X_test_5 = X_test_5.to_numpy()
X_test_7 = X_test_7.to_numpy()
X_test_8 = X_test_8.to_numpy()
X_test_9 = X_test_9.to_numpy()
X_test_10 = X_test_10.to_numpy()
X_test_5_pca = X_test_5_pca.to_numpy()
X_test_7_pca = X_test_7_pca.to_numpy()
X_test_8_pca = X_test_8_pca.to_numpy()
X_test_9_pca = X_test_9_pca.to_numpy()
X_test_10_pca = X_test_10_pca.to_numpy()
y_test_5 = y_test_5.to_numpy()
y_test_7 = y_test_7.to_numpy()
y_test_8 = y_test_8.to_numpy()
y_test_9 = y_test_9.to_numpy()
y_test_10 = y_test_10.to_numpy()

# Defining a function to compute test score of the Model
def test_scores(model):
  score5 = model.evaluate(X_test_5, y_test_5)
  score7 = model.evaluate(X_test_7, y_test_7)
  score8 = model.evaluate(X_test_8, y_test_8)
  score9 = model.evaluate(X_test_9, y_test_9)
  score10 = model.evaluate(X_test_10, y_test_10)
  test_scores = [score5[1],score7[1],score8[1],score9[1],score10[1]]
  
  return test_scores

# Validate the model MLP_3(Multi Layer Percpetron with Dropout and Weights) on Fold 5 test dataset to determine generalization
loss_mlp_3_test_5, acc_mlp_3_test_5 = mlp_3.evaluate(X_test_5, y_test_5, verbose=0)
print("\nFold 5 Test accuracy: %.1f%%" % (100.0 * acc_mlp_3_test_5))
print("\nFold 5 Test loss: %f" % (loss_mlp_3_test_5))

# Validate the model on Fold 7 test dataset to determine generalization
loss_mlp_3_test_7, acc_mlp_3_test_7 = mlp_3.evaluate(X_test_7, y_test_7, verbose=0)
print("\nFold 7 Test accuracy: %.1f%%" % (100.0 * acc_mlp_3_test_7))
print("\nFold 7 Test loss: %f" % (loss_mlp_3_test_7))

# Validate the model on Fold 8 test dataset to determine generalization
loss_mlp_3_test_8, acc_mlp_3_test_8 = mlp_3.evaluate(X_test_8, y_test_8, verbose=0)
print("\nFold 8 Test accuracy: %.1f%%" % (100.0 * acc_mlp_3_test_8))
print("\nFold 8 Test loss: %f" % (loss_mlp_3_test_8))

# Validate the model on Fold 9 test dataset to determine generalization
loss_mlp_3_test_9, acc_mlp_3_test_9 = mlp_3.evaluate(X_test_9, y_test_9, verbose=0)
print("\n Fold 9 Test accuracy: %.1f%%" % (100.0 * acc_mlp_3_test_9))
print("\nFold 9 Test loss: %f" % (loss_mlp_3_test_9))

# Validate the model on Fold 10 test dataset to determine generalization
loss_mlp_3_test_10, acc_mlp_3_test_10 = mlp_3.evaluate(X_test_10, y_test_10, verbose=0)
print("\nFold 10 Test accuracy: %.1f%%" % (100.0 * acc_mlp_3_test_10))
print("\nFold 10 Test loss: %f" % (loss_mlp_3_test_10))

# Compute Test Score of Multi Layer Perceptron with Dropouts and Weights
test_scores_mlp_3 = test_scores(mlp_3)
test_scores_mlp_3

type(test_scores_mlp_3)

# Print the mean & standard deviation of the Multi Layer Perceptron with Dropouts and Weights
Mean_mlp_3 = np.mean(test_scores_mlp_3)
std_mlp_3 = np.std(test_scores_mlp_3)
print(Mean_mlp_3)
print(std_mlp_3)

print(history_mlp_3.history.keys())

# summarize history for accuracy
plt.plot(history_mlp_3.history['sparse_categorical_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history_mlp_3.history['loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# Make class predictions Test datasets with the model MLP with Dropouts and Weights
y_pred5 = np.argmax(mlp_3.predict(X_test_5), axis=-1)
y_pred7 = np.argmax(mlp_3.predict(X_test_7), axis=-1)
y_pred8 = np.argmax(mlp_3.predict(X_test_8), axis=-1)
y_pred9 = np.argmax(mlp_3.predict(X_test_9), axis=-1)
y_pred10 = np.argmax(mlp_3.predict(X_test_10), axis=-1)

# Confusion Matrix for each testing Fold
cm_mlp3_test5 = confusion_matrix(y_test_5, y_pred5)
cm_mlp3_test7 = confusion_matrix(y_test_7, y_pred7)
cm_mlp3_test8 = confusion_matrix(y_test_8, y_pred8)
cm_mlp3_test9 = confusion_matrix(y_test_9, y_pred9)
cm_mlp3_test10 = confusion_matrix(y_test_10, y_pred10)

# Normalise the Confusion Matrix
cmn_mlp3_test5 = cm_mlp3_test5.astype('float')/cm_mlp3_test5.sum(axis=1)[:, np.newaxis]
cmn_mlp3_test7 = cm_mlp3_test7.astype('float')/cm_mlp3_test7.sum(axis=1)[:, np.newaxis]
cmn_mlp3_test8 = cm_mlp3_test8.astype('float')/cm_mlp3_test8.sum(axis=1)[:, np.newaxis]
cmn_mlp3_test9 = cm_mlp3_test9.astype('float')/cm_mlp3_test9.sum(axis=1)[:, np.newaxis]
cmn_mlp3_test10 = cm_mlp3_test10.astype('float')/cm_mlp3_test10.sum(axis=1)[:, np.newaxis]

import seaborn as sns
fig = plt.figure(figsize=(8,6))
ax= plt.subplot()

ax = sns.heatmap(cmn_mlp3_test5, annot=True, fmt='.2%', cmap='Blues')

ax.set_title('Seaborn Confusion Matrix with labels\n\n');
ax.set_xlabel('\nPredicted Values')
ax.set_ylabel('Actual Values ');

## Ticket labels - List must be in alphabetical order
ax.xaxis.set_ticklabels(['0','1','2', '3', '4', '5', '6', '7', '8', '9'])
ax.yaxis.set_ticklabels(['0','1','2', '3', '4', '5', '6', '7', '8', '9'])

## Display the visualization of the Confusion Matrix.
plt.show()

import seaborn as sns
fig = plt.figure(figsize=(8,6))
ax= plt.subplot()

ax = sns.heatmap(cmn_mlp3_test7, annot=True, fmt='.2%', cmap='Blues')

ax.set_title('Seaborn Confusion Matrix with labels\n\n');
ax.set_xlabel('\nPredicted Values')
ax.set_ylabel('Actual Values ');

## Ticket labels - List must be in alphabetical order
ax.xaxis.set_ticklabels(['0','1','2', '3', '4', '5', '6', '7', '8', '9'])
ax.yaxis.set_ticklabels(['0','1','2', '3', '4', '5', '6', '7', '8', '9'])

## Display the visualization of the Confusion Matrix.
plt.show()

import seaborn as sns
fig = plt.figure(figsize=(8,6))
ax= plt.subplot()

ax = sns.heatmap(cmn_mlp3_test8, annot=True, fmt='.2%', cmap='Blues')

ax.set_title('Seaborn Confusion Matrix with labels\n\n');
ax.set_xlabel('\nPredicted Values')
ax.set_ylabel('Actual Values ');

## Ticket labels - List must be in alphabetical order
ax.xaxis.set_ticklabels(['0','1','2', '3', '4', '5', '6', '7', '8', '9'])
ax.yaxis.set_ticklabels(['0','1','2', '3', '4', '5', '6', '7', '8', '9'])

## Display the visualization of the Confusion Matrix.
plt.show()

import seaborn as sns
fig = plt.figure(figsize=(8,6))
ax= plt.subplot()

ax = sns.heatmap(cmn_mlp3_test9, annot=True, fmt='.2%', cmap='Blues')

ax.set_title('Seaborn Confusion Matrix with labels\n\n');
ax.set_xlabel('\nPredicted Values')
ax.set_ylabel('Actual Values ');

## Ticket labels - List must be in alphabetical order
ax.xaxis.set_ticklabels(['0','1','2', '3', '4', '5', '6', '7', '8', '9'])
ax.yaxis.set_ticklabels(['0','1','2', '3', '4', '5', '6', '7', '8', '9'])

## Display the visualization of the Confusion Matrix.
plt.show()

import seaborn as sns
fig = plt.figure(figsize=(8,6))
ax= plt.subplot()

ax = sns.heatmap(cmn_mlp3_test10, annot=True, fmt='.2%', cmap='Blues')

ax.set_title('Seaborn Confusion Matrix with labels\n\n');
ax.set_xlabel('\nPredicted Values')
ax.set_ylabel('Actual Values ');

## Ticket labels - List must be in alphabetical order
ax.xaxis.set_ticklabels(['0','1','2', '3', '4', '5', '6', '7', '8', '9'])
ax.yaxis.set_ticklabels(['0','1','2', '3', '4', '5', '6', '7', '8', '9'])

## Display the visualization of the Confusion Matrix.
plt.show()

# Classification Report Fold 5 
from sklearn.metrics import classification_report
print('\nClassification Report of ANN\n')
print(classification_report(y_test_5,y_pred5, target_names=['0','1','2', '3', '4', '5', '6', '7', '8', '9']))

# Classification Report Fold 7
from sklearn.metrics import classification_report
print('\nClassification Report of ANN\n')
print(classification_report(y_test_7,y_pred7, target_names=['0','1','2', '3', '4', '5', '6', '7', '8', '9']))

# Classification Report Fold 8
from sklearn.metrics import classification_report
print('\nClassification Report of ANN\n')
print(classification_report(y_test_8,y_pred8, target_names=['0','1','2', '3', '4', '5', '6', '7', '8', '9']))

# Classification Report Fold 9
from sklearn.metrics import classification_report
print('\nClassification Report of ANN\n')
print(classification_report(y_test_9,y_pred9, target_names=['0','1','2', '3', '4', '5', '6', '7', '8', '9']))

# Classification Report Fold 10
from sklearn.metrics import classification_report
print('\nClassification Report of ANN\n')
print(classification_report(y_test_10,y_pred10, target_names=['0','1','2', '3', '4', '5', '6', '7', '8', '9']))

# validate the model on test dataset to determine generalization
loss_mlp_3pca_test_5, acc_mlp_3pca_test_5 = mlp_3_pca.evaluate(X_test_5_pca, y_test_5, verbose=0)
print("\nFold 5 Test accuracy: %.1f%%" % (100.0 * acc_mlp_3pca_test_5))
print("\nFold 5 Test loss: %.1f%%" % (loss_mlp_3pca_test_5))

# validate the model on test dataset to determine generalization
loss_mlp_3pca_test_7, acc_mlp_3pca_test_7 = mlp_3_pca.evaluate(X_test_7_pca, y_test_7, verbose=0)
print("\nFold 7 Test accuracy: %.1f%%" % (100.0 * acc_mlp_3pca_test_7))
print("\nFold 7 Test loss: %.1f%%" % (loss_mlp_3pca_test_7))

# validate the model on test dataset to determine generalization
loss_mlp_3pca_test_8, acc_mlp_3pca_test_8 = mlp_3_pca.evaluate(X_test_8_pca, y_test_8, verbose=0)
print("\nFold 8 Test accuracy: %.1f%%" % (100.0 * acc_mlp_3pca_test_8))
print("\nFold 8 Test loss: %f" % (loss_mlp_3pca_test_8))

# validate the model on test dataset to determine generalization
loss_mlp_3pca_test_9, acc_mlp_3pca_test_9 = mlp_3_pca.evaluate(X_test_9_pca, y_test_9, verbose=0)
print("\nFold 9 Test accuracy: %.1f%%" % (100.0 * acc_mlp_3pca_test_9))
print("\nFold 9 Test loss: %f" % (loss_mlp_3pca_test_9))

# validate the model on test dataset to determine generalization
loss_mlp_3pca_test_10, acc_mlp_3pca_test_10 = mlp_3_pca.evaluate(X_test_10_pca, y_test_10, verbose=0)
print("\nFold 10 Test accuracy: %.1f%%" % (100.0 * acc_mlp_3pca_test_10))
print("\nFold 10 Test loss: %f" % (loss_mlp_3pca_test_10))

def test_scores_pca(model):
  score5 = model.evaluate(X_test_5_pca, y_test_5)
  score7 = model.evaluate(X_test_7_pca, y_test_7)
  score8 = model.evaluate(X_test_8_pca, y_test_8)
  score9 = model.evaluate(X_test_9_pca, y_test_9)
  score10 = model.evaluate(X_test_10_pca, y_test_10)
  test_scores_pca = [score5[1],score7[1],score8[1],score9[1],score10[1]]
  
  return test_scores_pca

testscore_pca = test_scores_pca(mlp_3_pca)
testscore_pca

print((np.mean(testscore_pca), np.std(testscore_pca)))

# Make class predictions with the model
y_pred5_mlp_3 = np.argmax(mlp_3_pca.predict(X_test_5_pca), axis=-1)
y_pred7_mlp_3 = np.argmax(mlp_3_pca.predict(X_test_7_pca), axis=-1)
y_pred8_mlp_3 = np.argmax(mlp_3_pca.predict(X_test_8_pca), axis=-1)
y_pred9_mlp_3 = np.argmax(mlp_3_pca.predict(X_test_9_pca), axis=-1)
y_pred10_mlp_3 = np.argmax(mlp_3_pca.predict(X_test_10_pca), axis=-1)

# Confusion Matrix
cm_mlp_3_pca_test5 = confusion_matrix(y_test_5, y_pred5_mlp_3)
cm_mlp_3_pca_test7 = confusion_matrix(y_test_7, y_pred7_mlp_3)
cm_mlp_3_pca_test8 = confusion_matrix(y_test_8, y_pred8_mlp_3)
cm_mlp_3_pca_test9 = confusion_matrix(y_test_9, y_pred9_mlp_3)
cm_mlp_3_pca_test10 = confusion_matrix(y_test_10, y_pred10_mlp_3)

# Normalise the Confusion Matrix
cmn_mlp_3_test5 = cm_mlp_3_pca_test5.astype('float')/cm_mlp_3_pca_test5.sum(axis=1)[:, np.newaxis]
cmn_mlp_3_test7 = cm_mlp_3_pca_test7.astype('float')/cm_mlp_3_pca_test7.sum(axis=1)[:, np.newaxis]
cmn_mlp_3_test8 = cm_mlp_3_pca_test8.astype('float')/cm_mlp_3_pca_test8.sum(axis=1)[:, np.newaxis]
cmn_mlp_3_test9 = cm_mlp_3_pca_test9.astype('float')/cm_mlp_3_pca_test9.sum(axis=1)[:, np.newaxis]
cmn_mlp_3_test10 = cm_mlp_3_pca_test10.astype('float')/cm_mlp_3_pca_test10.sum(axis=1)[:, np.newaxis]

import seaborn as sns
fig = plt.figure(figsize=(16, 14))
ax= plt.subplot()

ax = sns.heatmap(cmn_mlp_3_test5, annot=True, fmt='.2%', cmap='Blues')

ax.set_title('Seaborn Confusion Matrix with labels\n\n');
ax.set_xlabel('\nPredicted Values')
ax.set_ylabel('Actual Values ');

## Ticket labels - List must be in alphabetical order
ax.xaxis.set_ticklabels(['0','1','2', '3', '4', '5', '6', '7', '8', '9'])
ax.yaxis.set_ticklabels(['0','1','2', '3', '4', '5', '6', '7', '8', '9'])

## Display the visualization of the Confusion Matrix.
plt.show()

import seaborn as sns
fig = plt.figure(figsize=(16, 14))
ax= plt.subplot()

ax = sns.heatmap(cmn_mlp_3_test7, annot=True, fmt='.2%', cmap='Blues')

ax.set_title('Seaborn Confusion Matrix with labels\n\n');
ax.set_xlabel('\nPredicted Values')
ax.set_ylabel('Actual Values ');

## Ticket labels - List must be in alphabetical order
ax.xaxis.set_ticklabels(['0','1','2', '3', '4', '5', '6', '7', '8', '9'])
ax.yaxis.set_ticklabels(['0','1','2', '3', '4', '5', '6', '7', '8', '9'])

## Display the visualization of the Confusion Matrix.
plt.show()

import seaborn as sns
fig = plt.figure(figsize=(16, 14))
ax= plt.subplot()

ax = sns.heatmap(cmn_mlp_3_test8, annot=True, fmt='.2%', cmap='Blues')

ax.set_title('Seaborn Confusion Matrix with labels\n\n');
ax.set_xlabel('\nPredicted Values')
ax.set_ylabel('Actual Values ');

## Ticket labels - List must be in alphabetical order
ax.xaxis.set_ticklabels(['0','1','2', '3', '4', '5', '6', '7', '8', '9'])
ax.yaxis.set_ticklabels(['0','1','2', '3', '4', '5', '6', '7', '8', '9'])

## Display the visualization of the Confusion Matrix.
plt.show()

import seaborn as sns
fig = plt.figure(figsize=(16, 14))
ax= plt.subplot()

ax = sns.heatmap(cmn_mlp_3_test9, annot=True, fmt='.2%', cmap='Blues')

ax.set_title('Seaborn Confusion Matrix with labels\n\n');
ax.set_xlabel('\nPredicted Values')
ax.set_ylabel('Actual Values ');

## Ticket labels - List must be in alphabetical order
ax.xaxis.set_ticklabels(['0','1','2', '3', '4', '5', '6', '7', '8', '9'])
ax.yaxis.set_ticklabels(['0','1','2', '3', '4', '5', '6', '7', '8', '9'])

## Display the visualization of the Confusion Matrix.
plt.show()

import seaborn as sns
fig = plt.figure(figsize=(16, 14))
ax= plt.subplot()

ax = sns.heatmap(cmn_mlp_3_test10, annot=True, fmt='.2%', cmap='Blues')

ax.set_title('Seaborn Confusion Matrix with labels\n\n');
ax.set_xlabel('\nPredicted Values')
ax.set_ylabel('Actual Values ');

## Ticket labels - List must be in alphabetical order
ax.xaxis.set_ticklabels(['0','1','2', '3', '4', '5', '6', '7', '8', '9'])
ax.yaxis.set_ticklabels(['0','1','2', '3', '4', '5', '6', '7', '8', '9'])

## Display the visualization of the Confusion Matrix.
plt.show()

# Classification Report
from sklearn.metrics import classification_report
print('\nClassification Report \n')
print(classification_report(y_test_5, y_pred5_mlp_3, target_names=['0','1','2', '3', '4', '5', '6', '7', '8', '9']))

# Classification Report
from sklearn.metrics import classification_report
print('\nClassification Report \n')
print(classification_report(y_test_7, y_pred7_mlp_3, target_names=['0','1','2', '3', '4', '5', '6', '7', '8', '9']))

# Classification Report
from sklearn.metrics import classification_report
print('\nClassification Report \n')
print(classification_report(y_test_8, y_pred8_mlp_3, target_names=['0','1','2', '3', '4', '5', '6', '7', '8', '9']))

# Classification Report
from sklearn.metrics import classification_report
print('\nClassification Report \n')
print(classification_report(y_test_9, y_pred9_mlp_3, target_names=['0','1','2', '3', '4', '5', '6', '7', '8', '9']))

# Classification Report
from sklearn.metrics import classification_report
print('\nClassification Report \n')
print(classification_report(y_test_10, y_pred10_mlp_3, target_names=['0','1','2', '3', '4', '5', '6', '7', '8', '9']))

"""## CNN"""

# Validate the model on test dataset to determine generalization
cnn_3_test5 = cnn_1.evaluate(X_test_img_5, y_test_img_5, verbose=0)
print("\n Fold 5 Test accuracy: %.1f%%" % (100.0 * cnn_3_test5[1]))
print("\nFold 5 Test loss: %f" % (cnn_3_test5[0]))

# Validate the model on test dataset to determine generalization
cnn_3_test7 = cnn_3.evaluate(X_test_img_7, y_test_img_7, verbose=0)
print("\nFold 7 Test accuracy: %.1f%%" % (100.0 * cnn_3_test7[1]))
print("\nFold 7 Test loss: %f" % (cnn_3_test7[0]))

# Validate the model on test dataset to determine generalization
cnn_3_test8 = cnn_3.evaluate(X_test_img_8, y_test_img_8, verbose=0)
print("\nFold 8 Test accuracy: %.1f%%" % (100.0 * cnn_3_test8[1]))
print("\nFold 8 Test loss: %f" % (cnn_3_test8[0]))

# Validate the model on test dataset to determine generalization
cnn_3_test9 = cnn_3.evaluate(X_test_img_9, y_test_img_9, verbose=0)
print("\nFold 9 Test accuracy: %.1f%%" % (100.0 * cnn_3_test9[1]))
print("\nFold 9 Test loss: %f" % (cnn_3_test9[0]))

# Validate the model on test dataset to determine generalization
cnn_3_test10 = cnn_3.evaluate(X_test_img_10, y_test_img_10, verbose=0)
print("\nFold 10 Test accuracy: %.1f%%" % (100.0 * cnn_3_test10[1]))
print("\nFold 10 Test loss: %f" % (cnn_3_test10[0]))

def test_scores_cnn(model):
  cnn_score5 = model.evaluate(X_test_img_5, y_test_img_5 )
  cnn_score7 = model.evaluate(X_test_img_7, y_test_img_7)
  cnn_score8 = model.evaluate(X_test_img_8, y_test_img_8)
  cnn_score9 = model.evaluate(X_test_img_9, y_test_img_9)
  cnn_score10 = model.evaluate(X_test_img_10, y_test_img_10)
  test_scores_cnn = [cnn_score5[1],cnn_score7[1],cnn_score8[1],cnn_score9[1],cnn_score10[1]]
  
  return test_scores_cnn

cnn_3_testscore = test_scores_cnn(cnn_3)

type(cnn_3_testscore)

print((np.mean(cnn_3_testscore), np.std(cnn_3_testscore)))

ypred5_cnn_3 = np.argmax(cnn_3.predict(X_test_img_5), axis=-1)
ypred7_cnn_3 = np.argmax(cnn_3.predict(X_test_img_7), axis=-1)
ypred8_cnn_3 = np.argmax(cnn_3.predict(X_test_img_8), axis=-1)
ypred9_cnn_3 = np.argmax(cnn_3.predict(X_test_img_9), axis=-1)
ypred10_cnn_3 = np.argmax(cnn_3.predict(X_test_img_10), axis=-1)

# Confusion Matrix CNN_3 on test datset
cm_cnn3_test5 = confusion_matrix(y_test_img_5, ypred5_cnn_3 )
cm_cnn3_test7 = confusion_matrix(y_test_img_7, ypred7_cnn_3 )
cm_cnn3_test8 = confusion_matrix(y_test_img_8, ypred8_cnn_3 )
cm_cnn3_test9 = confusion_matrix(y_test_img_9, ypred9_cnn_3 )
cm_cnn3_test10 = confusion_matrix(y_test_img_10, ypred10_cnn_3 )

# Normalise the Confusion Matrix of each testing fold
cmn_test5 = cm_cnn3_test5.astype('float')/cm_cnn3_test5.sum(axis=1)[:, np.newaxis]
cmn_test7 = cm_cnn3_test7.astype('float')/cm_cnn3_test7.sum(axis=1)[:, np.newaxis]
cmn_test8 = cm_cnn3_test8.astype('float')/cm_cnn3_test8.sum(axis=1)[:, np.newaxis]
cmn_test9 = cm_cnn3_test9.astype('float')/cm_cnn3_test9.sum(axis=1)[:, np.newaxis]
cmn_test10 = cm_cnn3_test10.astype('float')/cm_cnn3_test10.sum(axis=1)[:, np.newaxis]

# Plot Confusion Matrix for each fold
import seaborn as sns
fig = plt.figure(figsize=(8,6))
ax= plt.subplot()

ax = sns.heatmap(cmn_test5, annot=True, fmt='.2%', cmap='Blues')
ax.set_title('Seaborn Confusion Matrix with labels\n\n');
ax.set_xlabel('\nPredicted Values')
ax.set_ylabel('Actual Values ');

## Ticket labels - List must be in alphabetical order
ax.xaxis.set_ticklabels(['0','1','2', '3', '4', '5', '6', '7', '8', '9'])
ax.yaxis.set_ticklabels(['0','1','2', '3', '4', '5', '6', '7', '8', '9'])

## Display the visualization of the Confusion Matrix.
plt.show()

# Plot Confusion Matrix for each fold
import seaborn as sns
fig = plt.figure(figsize=(8,6))
ax= plt.subplot()

ax = sns.heatmap(cmn_test7, annot=True, fmt='.2%', cmap='Blues')
ax.set_title('Seaborn Confusion Matrix with labels\n\n');
ax.set_xlabel('\nPredicted Values')
ax.set_ylabel('Actual Values ');

## Ticket labels - List must be in alphabetical order
ax.xaxis.set_ticklabels(['0','1','2', '3', '4', '5', '6', '7', '8', '9'])
ax.yaxis.set_ticklabels(['0','1','2', '3', '4', '5', '6', '7', '8', '9'])

## Display the visualization of the Confusion Matrix.
plt.show()

# Plot Confusion Matrix for each fold
import seaborn as sns
fig = plt.figure(figsize=(8,6))
ax= plt.subplot()

ax = sns.heatmap(cmn_test8, annot=True, fmt='.2%', cmap='Blues')
ax.set_title('Seaborn Confusion Matrix with labels\n\n');
ax.set_xlabel('\nPredicted Values')
ax.set_ylabel('Actual Values ');

## Ticket labels - List must be in alphabetical order
ax.xaxis.set_ticklabels(['0','1','2', '3', '4', '5', '6', '7', '8', '9'])
ax.yaxis.set_ticklabels(['0','1','2', '3', '4', '5', '6', '7', '8', '9'])

## Display the visualization of the Confusion Matrix.
plt.show()

# Plot Confusion Matrix for each fold
import seaborn as sns
fig = plt.figure(figsize=(8,6))
ax= plt.subplot()

ax = sns.heatmap(cmn_test9, annot=True, fmt='.2%', cmap='Blues')
ax.set_title('Seaborn Confusion Matrix with labels\n\n');
ax.set_xlabel('\nPredicted Values')
ax.set_ylabel('Actual Values ');

## Ticket labels - List must be in alphabetical order
ax.xaxis.set_ticklabels(['0','1','2', '3', '4', '5', '6', '7', '8', '9'])
ax.yaxis.set_ticklabels(['0','1','2', '3', '4', '5', '6', '7', '8', '9'])

## Display the visualization of the Confusion Matrix.
plt.show()

# Plot Confusion Matrix for each fold
import seaborn as sns
fig = plt.figure(figsize=(8,6))
ax= plt.subplot()

ax = sns.heatmap(cmn_test10, annot=True, fmt='.2%', cmap='Blues')
ax.set_title('Seaborn Confusion Matrix with labels\n\n');
ax.set_xlabel('\nPredicted Values')
ax.set_ylabel('Actual Values ');

## Ticket labels - List must be in alphabetical order
ax.xaxis.set_ticklabels(['0','1','2', '3', '4', '5', '6', '7', '8', '9'])
ax.yaxis.set_ticklabels(['0','1','2', '3', '4', '5', '6', '7', '8', '9'])

## Display the visualization of the Confusion Matrix.
plt.show()

# Classification Report
from sklearn.metrics import classification_report
print('\nClassification Report \n')
print(classification_report(y_test_img_5, ypred5_cnn_3, target_names=['0','1','2', '3', '4', '5', '6', '7', '8', '9']))

# Classification Report
from sklearn.metrics import classification_report
print('\nClassification Report \n')
print(classification_report(y_test_img_7, ypred7_cnn_3, target_names=['0','1','2', '3', '4', '5', '6', '7', '8', '9']))

# Classification Report
from sklearn.metrics import classification_report
print('\nClassification Report \n')
print(classification_report(y_test_img_8, ypred8_cnn_3, target_names=['0','1','2', '3', '4', '5', '6', '7', '8', '9']))

# Classification Report
from sklearn.metrics import classification_report
print('\nClassification Report \n')
print(classification_report(y_test_img_9, ypred9_cnn_3, target_names=['0','1','2', '3', '4', '5', '6', '7', '8', '9']))

# Classification Report
from sklearn.metrics import classification_report
print('\nClassification Report \n')
print(classification_report(y_test_img_10, ypred10_cnn_3, target_names=['0','1','2', '3', '4', '5', '6', '7', '8', '9']))

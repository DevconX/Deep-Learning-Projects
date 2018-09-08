# Speech-to-Text
Gather speech-to-text models on Tensorflow

## Requirements
  * NumPy >= 1.11.1
  * TensorFlow >= 1.3
  * librosa
  * tqdm
  * matplotlib
  * scipy
  * Python >= 3.5

## Data
<img src="https://lh6.ggpht.com/eQewbtpIdf19JW49AyJmRbUB0RkmoDCsi4j-g9Uk_KiffBOSZ-0IyhMr6lgoBimeXPk=w300-rw" height="300" align="right">

I trained these models on [this dataset, LJ Speech Dataset](https://keithito.com/LJ-Speech-Dataset/)

## How to train
1. You must download the dataset first, extract to any directory.
2. Change directory location in utils.py.
```python
# change here
data = "/home/husein/space/tacotron/LJSpeech-1.0/"
sampledir = 'samples'
```
3. If you want faster data load, run caching.py.
4. Run train.ipynb using jupyter notebook.


## Models covered
1. Tacotron

### *This repository will update overtime*

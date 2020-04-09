# OpenNLU

OpenNLU is a front-end interface for Natural Language Understanding (NLU) development, powered by RASA, Tensorflow and PyTorch.

![home](readme_images/home.JPG "home")

## Features

- Data Management
  - Editor for MD, JSON files (intents, entities) in Rasa format
  - Editor for labels, seq.in, seq.out files in Goo et al format
  - Conversion between the two above formats
  - Split dataset
  - Merge dataset
  
  ![rasa_data](readme_images/rasa_data.JPG "rasa_data")
  
  ![rasa_split](readme_images/rasa_split.JPG "rasa_split")
  
  ![pt_tf_data](readme_images/pt_tf_data.JPG "pt_tf_data")
  
  ![pt_tf_editor](readme_images/pt_tf_editor.JPG "pt_tf_editor")
  
- Training
  - Train Rasa models
  - Train BERT model for Joint Intent Classification and Slot Filling (Tensorflow / PyTorch)
  
  ![train_home](readme_images/train_home.JPG "train_home")
  
  ![rasa_train](readme_images/rasa_train.JPG "rasa_train")
  
  ![tf_train](readme_images/tf_train.JPG "tf_train")
  
- Load
  - Loads trained models (including models trained outside of OpenNLU)
  
  ![load_model](readme_images/load_model.JPG "load_model")
  
- Evaluation
  - Metrics: acc, f1, recall, precision
  - Intent Confusion Matrix
  - Confidence Histogram
  
  ![evaluate](readme_images/evaluate.JPG "evaluate")
  
- Predict
  - Return metrics and prediction for single sentence

## Run

```bash
$ python3 run.py
```

## Credits for Tensorflow, PyTorch backend implementation

- [dialog-nlu](https://github.com/MahmoudWahdan/dialog-nlu)
- [JointBERT](https://github.com/monologg/JointBERT)

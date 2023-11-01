# Digit Recognizer

## Contents

- [Download Dataset](#download-dataset)
- [Model Training & Testing](#model-training-&-testing-spam-classification)
- [Serialization](#serialization)

## Note:

- All files or commands should be executed from the repo home directory.

### Model Training & Testing

Trains & tests the classification model using Kaggle's Digit Recognizer dataset.

#### Arguments

- **--model_version or -mv**: Version by which the trained model files should be saved as.

#### Sample Usage

```
python3 src/digit_recognizer/train.py --model_version 1.0.0
```

or

```
python3 src/digit_recognizer/train.py -mv 1.0.0
```

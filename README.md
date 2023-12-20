# sleep_classification

sleep/awake classifier.

Code is inspired and partly copied from
<https://github.com/ojwalch/sleep_classifiers/>

## Installation

```bash
pip install git+https://github.com/enrilohm/sleep_classification.git
```
## Usage

### Inference

```python
#Python
from sleep_classification import SleepClassifier
SleepClassifier.predict(heart_rate_df, accerlation_df)
```

### Training and Evaluation

For messing around with the models you might want to clone the repo and make use of the Makefile
```bash
git clone https://github.com/enrilohm/sleep_classification.git
```

install dependencies
```bash
make dep
```

download the data from physionet
```bash
make data
```

and train the models yourself
```bash
make train
```

Models are saved to the models directory. To use your own models for inference, put them into sleep_classification/data.

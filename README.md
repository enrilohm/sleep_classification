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
import pandas as pd
import random
heart_rate_df = pd.DataFrame([(i*10**9*50, 60 + random.randint(0,2)) for i in range(100000//50)]).set_index(0)
acceleration_df = pd.DataFrame([(i*10**9/50, 0,0, -9.81) for i in range(5000000)]).set_index(0)


sleep_classifier = SleepClassifier()
prediction = sleep_classifier.predict(heart_rate_df, acceleration_df)
```
The predict method expects a single column heart rate DataFrame and a 3 column acceleration DataFrame in SI units. For both DataFrames, the index should be datetime information compatible with pandas.to_datetime(). In this example, the datetime index is in nanoseconds.

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

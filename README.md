# sleep_classification
sleep/awake classifier.

Code is inspired and partly copied from
https://github.com/ojwalch/sleep_classifiers/
also

## Installation
```bash
pip install git+https://github.com/enrilohm/sleep_classification.git
```

## Usage

```python
#Python
from sleep_classification import SleepClassifier
SleepClassifier.predict(heart_rate_df, accerlation_df)
```

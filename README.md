# 🏆 Kaggle: Hedge Fund Time Series Forecasting

## Team Members
- [Le Gia Bao]
- [Hoang Duc Bao]

## Competition
[Hedge Fund - Time Series Forecasting](https://kaggle.com/competitions/ts-forecasting)

## Project Structure
```
├── notebooks/          # Kaggle notebooks
├── src/                # Reusable Python modules
├── submissions/        # CSV submission files
├── docs/               # Documentation & logs
├── requirements.txt    # Dependencies
└── .gitignore
```

## Quick Start on Kaggle

```python
# In your Kaggle notebook, Cell 1:
!git clone https://github.com/hoangducbao/-n-CS116.git /kaggle/working/repo

# Cell 2:
import sys
sys.path.append('/kaggle/working/repo/src')

from config import *
from data_loader import load_data
from features import engineer_features
from models import train_per_horizon, predict_per_horizon, create_submission
from evaluation import weighted_rmse_score
```

## Local Setup
```bash
pip install -r requirements.txt
```

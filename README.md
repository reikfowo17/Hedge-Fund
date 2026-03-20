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

## 🚀 Hướng Dẫn Chạy & Kiểm Thử Trên Kaggle

### Bước 1: Kéo Code Từ GitHub Về
Tạo một Notebook mới trên Kaggle. Thêm GitHub Token vào **Add-ons → Secrets** (label: `GITHUB_TOKEN`).
```python
# Cell 1: Clone repo (Private, dùng Kaggle Secrets)
from kaggle_secrets import UserSecretsClient
import os, sys

secrets = UserSecretsClient()
github_token = secrets.get_secret("GITHUB_TOKEN")

os.system("rm -rf /kaggle/working/repo")
os.system(f"git clone https://{github_token}@github.com/reikfowo17/CS116.git /kaggle/working/repo")

sys.path.append('/kaggle/working/repo/src')
```

### Bước 2: Import Modules & Load Dữ Liệu
Hãy đảm bảo phần **Data** đã thêm tập dataset chứa `train.parquet` và `test.parquet`.
```python
# Cell 2: Import & Load data
from config import *
from data_loader import load_data

train_df, test_df = load_data(reduce_memory=True)
```

### Bước 3: Huấn Luyện & Dự Đoán

```python
# Cell 3: Train & Predict
from models import train_and_predict_all_horizons, create_submission

sub_clip, sub_raw, scores = train_and_predict_all_horizons()
```

### Bước 4: Xuất File Submission
```python
# Cell 4: Tạo file submission.csv
create_submission(sub_clip, "submission.csv")

# Copy ra thư mục gốc để Kaggle nhận Output
sub_clip.to_csv("/kaggle/working/submission.csv", index=False)
```

## 💻 Local Setup
```bash
pip install -r requirements.txt
```

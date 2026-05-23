# Helmet Detection Kaggle Dataset

Source: Kaggle dataset `andrewmvd/helmet-detection`, downloaded with `kagglehub`.

Reference page: https://www.kaggle.com/datasets/andrewmvd/helmet-detection

This dataset contains 764 images and 764 PASCAL VOC annotation files for two object detection classes:

- `With Helmet`
- `Without Helmet`

Local structure:

```text
test-data/helmet-detection-kaggle/
├── images/
│   └── BikesHelmets*.png
└── annotations/
    └── BikesHelmets*.xml
```

Run dataset evaluation:

```powershell
C:\Program Files\Python311\python.exe scripts\evaluate_dataset.py
```

Outputs:

- `output/test-results/dataset_evaluation_details.csv`
- `output/test-results/dataset_evaluation_summary.json`

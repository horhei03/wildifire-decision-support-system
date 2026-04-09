# Wildfire Crew Extraction Warning System

**Deep Learning Decision Support System for Wildfire Safety**

## Overview

This project implements a decision support system that predicts wildfire danger to firefighting crews using Convolutional Long Short-Term Memory (ConvLSTM) neural networks. By analysing FARSITE wildfire simulation outputs, the system provides 10-minute advance warnings for crew extraction.

## Key Features

- **Spatiotemporal Deep Learning**: ConvLSTM architecture captures fire spread dynamics
- **Multi-Crew Prediction**: Simultaneous danger assessment for 6 crew positions
- **Safety-Focused Evaluation**: Optimised to minimise false negatives (missed dangers)
- **Geospatial Analysis**: Processes 8 km x 8 km domains at 5 m resolution
- **Interactive Dashboard**: Streamlit web interface for real-time monitoring

## Installation

```bash
python -m venv .venv
.venv\Scripts\activate   # Windows
pip install -r requirements.txt
```

## Usage

### 1. Generate Training Data

```bash
python scripts/generate_dataset.py \
    --data-config configs/data_config.yaml \
    --crew-config configs/crew_positions.yaml \
    --output-dir data/processed
```

### 2. Train Model

```bash
python scripts/train_model.py \
    --model-config configs/model_config.yaml \
    --training-config configs/training_config.yaml \
    --data-dir data/processed \
    --output-dir models/trained
```

### 3. Evaluate

```bash
python scripts/evaluate_model.py \
    --model-path models/trained/best_model.weights.h5 \
    --data-dir data/processed \
    --output-dir results
```

### 4. Launch Dashboard

```bash
streamlit run app/streamlit_app.py
```

## Project Structure

```
wildfire_dss/
├── src/
│   ├── preprocessing/     # FARSITE parsing, label generation, tensor building
│   ├── models/            # ConvLSTM architecture
│   ├── training/          # Training loop and callbacks
│   ├── evaluation/        # Metrics (precision, recall, F1, AUC, safety metrics)
│   └── inference/         # Prediction engine with severity classification
├── app/                   # Streamlit dashboard
├── configs/               # YAML configurations (model, training, data, crew positions)
├── scripts/               # Dataset generation, training, evaluation, HP tuning
├── figures/               # Thesis figures and tables
├── tests/                 # Unit tests
└── results/               # Test metrics
```

## Dataset

- **Source**: FARSITE wildfire simulation outputs
- **Patches**: 8 fuel types (forest, shrub, grass, mixed)
- **Scenarios**: 6 weather conditions per patch (wind speed/direction, fuel moisture)
- **Total**: 48 simulations (8 patches x 6 scenarios), 1,873 training sequences
- **Domain**: 8 km x 8 km at 5 m resolution (1600 x 1600 cells)
- **Input**: 10 timesteps of flame length + rate of spread (shape: 10, 320, 320, 2)

## Model Architecture

```
Input (10, 320, 320, 2)
  -> TimeDistributed AveragePooling2D (5x5)
  -> ConvLSTM2D (64 filters, 3x3) + BatchNorm + Dropout
  -> ConvLSTM2D (32 filters, 3x3) + BatchNorm + Dropout
  -> ConvLSTM2D (16 filters, 3x3) + BatchNorm + Dropout
  -> GlobalAveragePooling2D
  -> Dense (128) + Dense (64)
  -> Dense (6, sigmoid)
```

- **Optimizer**: Adam (lr=1e-4)
- **Loss**: Binary crossentropy
- **Metrics**: Accuracy, Precision, Recall, AUC-ROC

## Test Results (6 crews, 331 samples)

| Metric    | Value  |
|-----------|--------|
| Accuracy  | 83.53% |
| Precision | 77.35% |
| Recall    | 39.78% |
| F1 Score  | 52.54% |
| AUC-ROC   | 0.8869 |

## Configuration

All parameters are in `configs/`:

- `data_config.yaml` - data sources, patches, scenarios, danger criteria
- `model_config.yaml` - architecture (filters, kernel size, dense units, dropout)
- `training_config.yaml` - epochs, batch size, learning rate, callbacks
- `crew_positions.yaml` - crew locations and safety parameters

## Requirements

- Python 3.9+
- TensorFlow 2.13+
- NumPy, Pandas, Rasterio, GeoPandas
- Streamlit, Plotly
- See `requirements.txt` for full list

## License

Academic use only. All rights reserved.

---

**Note**: This is a research prototype for academic purposes. Not intended for operational wildfire management without further validation and testing.

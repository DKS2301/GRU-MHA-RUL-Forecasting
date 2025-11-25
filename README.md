# Battery RUL Prediction using GRU-MHA Model

Implementation of the research paper: **"A Hybrid GRU-MHA model for accurate battery RUL forecasting with feature selection"** by Abeer Aljohani & Saad Aljohani (2025).

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
- [Dataset Requirements](#dataset-requirements)
- [Usage](#usage)
- [Model Architecture](#model-architecture)
- [Results](#results)
- [Citation](#citation)
- [License](#license)

## 🎯 Overview

This implementation provides a hybrid deep learning approach combining **Gated Recurrent Units (GRU)** and **Multi-Head Attention (MHA)** mechanisms for accurate battery Remaining Useful Life (RUL) and State of Health (SoH) forecasting.

### Key Achievements (as reported in paper):
- **NMC-LCO 18650 Battery**: 0.002 MAE, 0.044 MSE, 99.99% R²
- **NASA B0005 Battery**: 0.005 MAE, 0.0706 MSE, 99.64% R²
- **NASA B0018 Battery**: 0.012 MAE, 0.109 MSE, 95.97% R²

## ✨ Features

- **Dual-Path Architecture**: Separate processing for correlated and uncorrelated features
- **Feature Selection**: Automatic Pearson correlation-based feature separation
- **Dynamic Swish Activation**: Custom activation function for improved forecasting
- **Ridge Regression**: Final stage regression for seasonal pattern prediction
- **Comprehensive Evaluation**: Comparison with ML baselines (KNN, SVR, Random Forest)
- **Visualization Tools**: Training curves, prediction plots, correlation matrices
- **Forward Feature Selection**: Identify most important features
- **Easy-to-Use Pipeline**: Single function call for complete workflow

## 📦 Requirements

```
python >= 3.8
numpy >= 1.19.0
pandas >= 1.2.0
matplotlib >= 3.3.0
seaborn >= 0.11.0
scikit-learn >= 0.24.0
tensorflow >= 2.8.0
```

## 🔧 Installation

### Option 1: Using pip

```bash
# Create virtual environment (recommended)
python -m venv battery_env
source battery_env/bin/activate  # On Windows: battery_env\Scripts\activate

# Install requirements
pip install numpy pandas matplotlib seaborn scikit-learn tensorflow

# For GPU support (optional but recommended)
pip install tensorflow-gpu
```

### Option 2: Using conda

```bash
# Create conda environment
conda create -n battery_env python=3.9
conda activate battery_env

# Install packages
conda install numpy pandas matplotlib seaborn scikit-learn
pip install tensorflow
```

## 📊 Dataset Requirements

### 1. NMC-LCO 18650 Battery Dataset

**Required Columns:**
- `Cycle-Index`: Battery cycle number
- `Discharge Time (s)`: Time for discharge in seconds
- `Decrease 3.6-3.4V (s)`: Voltage decrease time
- `Max Discharge Voltage (V)`: Maximum voltage during discharge
- `Min Charging Voltage (V)`: Minimum voltage during charging
- `Time at 4.15V (s)`: Time spent at 4.15V
- `Time at Constant Current (s)`: Constant current charging time
- `Charging Time (s)`: Total charging time
- `RUL`: Remaining Useful Life (target variable)

**Dataset Info:**
- 15,064 entries
- Source: Hawaii Natural Energy Institute
- 80% threshold for RUL calculation

### 2. NASA Battery Dataset (B0005, B0006, B0007, B0018)

**Required Columns:**
- `Cycle`: Cycle number
- `Time Measured(Sec)`: Measured time in seconds
- `Voltage Measured(V)`: Measured voltage
- `Current Measured`: Measured current
- `Temperature Measured`: Measured temperature
- `Capacity(Ah)`: Battery capacity
- `Signal Energy`: Energy signal
- `Fluctuation Index`: Fluctuation measurement
- `Skewness Index`: Skewness measurement
- `Kurtosis Index`: Kurtosis measurement

**Dataset Info:**
- Source: NASA Ames Prognostics Center
- Operating temperature: 24-25°C
- Charge rate: 1.5A constant current
- Discharge rate: 2A constant current

**SoH Calculation:**
```
SoH = Current_Capacity / Initial_Capacity
```

### Data Sources:

1. **NMC-LCO 18650**: Available from Hawaii Natural Energy Institute
2. **NASA Batteries**: Available from [NASA Prognostics Data Repository](https://ti.arc.nasa.gov/tech/dash/groups/pcoe/prognostic-data-repository/)

## 🚀 Usage

### Quick Start

```python
import jupyter notebook
# Open the .ipynb file

# For NMC-LCO 18650 Battery
results = run_complete_pipeline(
    data_path='data/nmc_lco_battery.csv',
    dataset_type='NMC',
    target_col='RUL',
    correlation_threshold=0.4,
    scale_factor=1.0
)

# For NASA Battery
results = run_complete_pipeline(
    data_path='data/nasa_B0005.csv',
    dataset_type='NASA',
    target_col='SoH',
    correlation_threshold=0.4,
    scale_factor=8.0  # Use 8.0 for values between 0-2
)

# Access results
print(f"Test R² Score: {results['test_metrics']['r2']:.4f}")
print(f"Test MAE: {results['test_metrics']['mae']:.6f}")
print(f"Test RMSE: {results['test_metrics']['rmse']:.6f}")

# Save trained model
results['model'].save('models/gru_mha_battery_model.h5')
```

### Step-by-Step Usage

```python
# 1. Load Data
df = load_nmc_lco_data('data/battery_data.csv')

# 2. Feature Selection
X = df.drop('RUL', axis=1)
y = df['RUL']
corr_features, uncorr_features, correlations = select_features_by_correlation(X, y, threshold=0.4)

# 3. Prepare Data
data_dict = prepare_data(df, 'RUL', corr_features, uncorr_features)

# 4. Build Model
model = build_gru_mha_model(
    correlated_dim=len(corr_features),
    uncorrelated_dim=len(uncorr_features),
    gru_units=4,
    mha_heads=8
)

# 5. Train Model
history = train_model(model, data_dict, epochs=250)

# 6. Apply Ridge Regression
train_pred, test_pred, ridge = apply_ridge_regression(model, data_dict, alpha=1.5)

# 7. Evaluate
test_metrics = evaluate_model(data_dict['y_test'], test_pred, data_dict['scaler_y'])
```

### Advanced Options

```python
# Customize hyperparameters
model = build_gru_mha_model(
    correlated_dim=2,
    uncorrelated_dim=6,
    gru_units=4,        # Number of GRU units
    dense_units=30,      # Dense layer neurons
    mha_heads=8,         # Multi-head attention heads
    mha_units=20,        # Attention key dimension
    scale_factor=1.0     # Dynamic swish scaling
)

# Custom training parameters
history = train_model(
    model, 
    data_dict, 
    epochs=250,          # Maximum epochs
    batch_size=32,       # Batch size
    patience=50          # Early stopping patience
)

# Custom ridge regression
_, test_pred, _ = apply_ridge_regression(
    model, 
    data_dict, 
    alpha=1.5           # Ridge regularization strength
)
```

## 🏗️ Model Architecture

### Overview

```
Input Features
    ├── Correlated Features Path
    │   ├── GRU Layer (4 units)
    │   ├── Attention Mechanism
    │   ├── Multi-Head Attention (8 heads)
    │   └── Flatten
    │
    └── Uncorrelated Features Path
        ├── GRU Layer (4 units)
        ├── Attention Mechanism
        ├── Multi-Head Attention (8 heads)
        └── Flatten
            ↓
        Concatenate
            ↓
        Dense (30 units, Swish)
            ↓
        Dense (15 units, Swish)
            ↓
        Dense (1 unit) + Dynamic Swish
            ↓
        Ridge Regression (α=1.5)
            ↓
        Final RUL/SoH Prediction
```

### Key Components

1. **Feature Separation**: 
   - Pearson correlation ≥ 0.4 → Correlated features
   - Pearson correlation < 0.4 → Uncorrelated features

2. **GRU Units**: Capture temporal dependencies in battery degradation

3. **Attention Mechanism**: Focus on important time steps

4. **Multi-Head Attention**: Parallel attention for different representation subspaces

5. **Dynamic Swish**: 
   ```
   f(x) = x / (s × (1 + e^(-x)))
   ```
   where s is the scaling factor

6. **Ridge Regression**: Final stage for seasonal pattern prediction

### Model Parameters

- Total Parameters: ~1,195
- Trainable Parameters: ~1,187
- Memory (256 samples): ~1.2 MB
- Inference Time (CPU): ~1.2s per sample
- Inference Time (GPU): ~0.01s per sample

## 📈 Results

### Performance Comparison

| Dataset | Model | MAE | MSE | RMSE | R² Score |
|---------|-------|-----|-----|------|----------|
| NMC-LCO | GRU-MHA+Ridge | 0.002 | 0.044 | - | 99.99% |
| NMC-LCO | CNN-LSTM | 0.032 | 0.178 | - | 93.64% |
| NMC-LCO | Random Forest | 0.005 | 0.071 | - | 99.98% |
| NASA B0005 | GRU-MHA+Ridge | 0.005 | 0.071 | 0.007 | 99.64% |
| NASA B0005 | CNN-LSTM | 0.027 | 0.164 | - | 76.04% |
| NASA B0006 | GRU-MHA+Ridge | 0.012 | 0.109 | 0.013 | 97.98% |
| NASA B0007 | GRU-MHA+Ridge | 0.012 | 0.105 | 0.015 | 97.49% |
| NASA B0018 | GRU-MHA+Ridge | 0.012 | 0.109 | 0.013 | 95.97% |

### Comparison with Literature

| Method | Dataset | RMSE | R² Score |
|--------|---------|------|----------|
| **GRU-MHA (Ours)** | NASA B0005 | **0.007** | **99.64%** |
| PSO-LSTM-Attention | NASA B0005 | 0.016 | - |
| DeTransformer | NASA B0005 | 0.080 | - |
| MLP | NASA B0005 | 0.066 | 93.34% |
| Temporal CNN | NASA B0005 | 0.018 | - |

## 🔍 Feature Importance

### NMC-LCO 18650 Battery
Most important features (in order):
1. Cycle-Index
2. Discharge Time (s)
3. Max Discharge Voltage (V)

### NASA Batteries
Most important features (in order):
1. Cycle
2. Time Measured (Sec)
3. Temperature Measured
4. Voltage Measured (V)

## 📊 Visualization Outputs

The implementation automatically generates:

1. **Correlation Matrix**: Heatmap of feature correlations
2. **RUL/SoH Trends**: Line plot showing degradation over cycles
3. **Training History**: Loss and MAE curves for training/validation
4. **Prediction Plots**: 
   - Time series: Actual vs Predicted
   - Scatter plot: Actual vs Predicted with diagonal reference
5. **Feature Selection**: Forward selection R² progression
6. **SHAP Values** (if enabled): Feature contribution to predictions

## 🛠️ Troubleshooting

### Common Issues

**1. ImportError: No module named 'tensorflow'**
```bash
pip install tensorflow
```

**2. Memory Error during training**
- Reduce batch_size: `history = train_model(model, data_dict, batch_size=16)`
- Use gradient checkpointing or reduce sequence length

**3. Poor performance on your dataset**
- Check data quality and preprocessing
- Adjust correlation threshold: Try 0.3-0.5 range
- Tune hyperparameters: GRU units, MHA heads, dense units
- Increase epochs or adjust learning rate

**4. Model not converging**
- Check feature scaling (should be normalized)
- Reduce learning rate: Use optimizer with lower LR
- Increase patience in EarlyStopping

**5. CSV Loading Errors**
- Ensure column names match exactly
- Check for missing values: `df.isnull().sum()`
- Verify data types: `df.dtypes`

## 🔬 Hyperparameter Tuning

### Recommended Ranges

| Parameter | Range | Default | Description |
|-----------|-------|---------|-------------|
| gru_units | 2-8 | 4 | GRU hidden units |
| mha_heads | 4-16 | 8 | Multi-head attention heads |
| mha_units | 10-40 | 20 | Attention key dimension |
| dense_units | 16-64 | 30 | Dense layer neurons |
| alpha (Ridge) | 0.1-10 | 1.5 | Ridge regularization |
| learning_rate | 1e-4 to 1e-2 | 0.001 | Optimizer learning rate |
| scale_factor | 1-8 | 1.0 or 8.0 | Dynamic swish scaling |

### Grid Search Example

```python
from sklearn.model_selection import ParameterGrid

param_grid = {
    'gru_units': [3, 4, 8],
    'mha_heads': [8, 16],
    'alpha': [0.1, 1.0, 1.5]
}

best_r2 = -np.inf
best_params = None

for params in ParameterGrid(param_grid):
    model = build_gru_mha_model(
        correlated_dim=len(corr_features),
        uncorrelated_dim=len(uncorr_features),
        gru_units=params['gru_units'],
        mha_heads=params['mha_heads']
    )
    # Train and evaluate...
    # Track best parameters
```

## 📝 Implementation Notes

### Design Decisions

1. **Two-Path Architecture**: Separates features by correlation to handle linear and non-linear relationships differently

2. **Dynamic Swish**: Allows flexible output scaling for different value ranges (0-2 vs 0-100s)

3. **Ridge Regression**: Addresses seasonal patterns and prevents overfitting in final predictions

4. **Nadam Optimizer**: Combines Nesterov momentum with Adam for faster convergence

5. **Early Stopping**: Prevents overfitting with 50-epoch patience

### Limitations

1. **Data Requirements**: Needs sufficient cycles for training (recommended: 100+ cycles)

2. **Sudden Changes**: May not capture abrupt capacity increases (recovery events)

3. **Feature Engineering**: Performance depends on quality of input features

4. **Computational Cost**: Training takes ~200-450 seconds depending on dataset size

### Future Enhancements

- GAN-based data augmentation for limited datasets
- Metaheuristic optimization (e.g., Gray Wolf Optimizer)
- Transfer learning between battery types
- Real-time monitoring dashboard
- Uncertainty quantification

## 🤝 Contributing

Contributions are welcome! Areas for improvement:

- Additional battery dataset support
- Bayesian hyperparameter optimization
- Ensemble methods
- Explainability tools (SHAP, LIME)
- Real-time deployment examples
- Docker containerization

## 📄 License

This implementation is provided as open-source for research purposes. The original paper is published in Energy Reports (Elsevier) under CC BY-NC-ND 4.0 license.

## 📚 Citation

If you use this implementation, please cite the original paper:

```bibtex
@article{aljohani2025hybrid,
  title={A Hybrid GRU-MHA model for accurate battery RUL forecasting with feature selection},
  author={Aljohani, Abeer and Aljohani, Saad},
  journal={Energy Reports},
  volume={14},
  pages={294--309},
  year={2025},
  publisher={Elsevier},
  doi={10.1016/j.egyr.2025.05.059}
}
```

## 📧 Contact

For questions or issues:
- Open an issue on GitHub
- Email: aahjohani@taibahu.edu.sa (Original Authors)

## 🙏 Acknowledgments

- Original authors: Abeer Aljohani (Taibah University) & Saad Aljohani (Western Michigan University)
- NASA Ames Prognostics Center for battery datasets
- Hawaii Natural Energy Institute for NMC-LCO battery data

---

**Last Updated**: November 2025  
**Version**: 1.0.0  
**Status**: Production Ready ✅

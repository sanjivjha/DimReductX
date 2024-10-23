# Automated Dimension Reduction Analysis

## Introduction

This package provides a comprehensive solution for automated dimension reduction analysis, combining feature engineering and various reduction techniques. It helps address the "curse of dimensionality" through intelligent feature selection and transformation.

### Key Concepts

1. **Dimension Reduction**
   - Process of reducing the number of features while preserving important information
   - Helps improve model performance, reduce overfitting, and speed up training
   - Essential for handling high-dimensional datasets

2. **Methods Implemented**
   - **PCA (Principal Component Analysis)**
     - Unsupervised technique
     - Creates new uncorrelated features
     - Maximizes variance explained
   
   - **Select K Best**
     - Supervised feature selection
     - Uses statistical tests (F-test, mutual information)
     - Maintains original feature interpretability
   
   - **Random Forest Feature Importance**
     - Model-based selection
     - Captures non-linear relationships
     - Robust feature ranking

3. **Automated Feature Engineering**
   - Generates derived features using mathematical transformations
   - Discovers complex relationships automatically
   - Includes logarithmic, polynomial, and interaction terms

## Installation

```bash
# Clone the repository
git clone https://github.com/sanjivjha/DimReductX.git
# Install required packages
pip install -r requirements.txt
```

## Quick Start

### Using Synthetic Data

python dimension_reduction/main.py

### Using Your Own Dataset

```python
import pandas as pd
from dimension_reduction.dimension_reduction import DimensionReducer

# Load your data
X = pd.read_csv('your_features.csv')
y = pd.read_csv('your_target.csv')  # or pd.Series for single target column

# Initialize reducer
reducer = DimensionReducer(
    task_type='regression',  # or 'classification'
    max_poly_degree=2  # maximum degree for polynomial features
)

# Perform reduction
optimized_datasets = reducer.reduce_dimensions(X, y)

# Generate report
report = reducer.generate_report()

# Access optimized datasets
for method, dataset in optimized_datasets.items():
    print(f"\nOptimized dataset using {method}:")
    print(dataset.shape)
    # Save to CSV if needed
    dataset.to_csv(f'optimized_{method}.csv', index=False)
```

### Getting DataFrame Output

```python
# Method 1: Direct access to optimized datasets
best_pca_df = optimized_datasets['pca']
best_random_forest_df = optimized_datasets['random_forest']

# Method 2: Get specific method with custom components
reducer = DimensionReducer(task_type='regression')
reducer.reduce_dimensions(X, y, 
                        methods=['pca'], 
                        n_components_list=[10])
custom_df = reducer.get_optimized_datasets()['pca']
```

## Input Data Requirements

Your input data should be:
- Features (X): pandas DataFrame
- Target (y): pandas Series or single-column DataFrame
- No missing values (handle these before using the package)
- Numeric data (encode categorical variables first)

Example input format:
```python
# Features DataFrame (X)
     feature1  feature2  feature3
0    1.2      0.5      3.1
1    2.3      1.1      4.2
...

# Target Series (y)
0    0
1    1
...
```

## Output Files

The package generates several outputs:

1. **PDF Report** (`analysis_report.pdf`)
   - Complete analysis summary
   - Feature importance rankings
   - Method comparisons
   - Detailed explanations

2. **Optimized Datasets** (`optimized_dataset_{method}.csv`)
   - Reduced datasets for best methods
   - Ready for model training

3. **Visualization Plots**
   - Correlation matrices
   - Performance comparisons
   - Feature importance plots

## Advanced Usage

### Customizing Analysis

```python
# Custom reduction with specific methods and components
reducer = DimensionReducer(task_type='regression')
optimized_datasets = reducer.reduce_dimensions(
    X, y,
    methods=['pca', 'random_forest'],
    n_components_list=[5, 10, 15]
)

# Access specific aspects of the analysis
feature_importance = reducer._get_feature_importance()
feature_stats = reducer._get_feature_statistics('feature_name')
```

### Handling Large Datasets

```python
# For very large datasets, you might want to limit feature generation
reducer = DimensionReducer(
    task_type='regression',
    max_poly_degree=1  # Limit polynomial features
)

# Or focus on specific methods
optimized_datasets = reducer.reduce_dimensions(
    X, y,
    methods=['random_forest'],  # Use only one method
    n_components_list=[10]      # Test only one size
)
```

## Common Issues and Solutions

1. **Memory Issues**
   - Reduce `max_poly_degree` to limit feature generation
   - Use fewer methods or component sizes
   - Process data in batches if possible

2. **Long Processing Time**
   - Start with a subset of your data to test
   - Limit the number of derived features
   - Focus on specific reduction methods

3. **Poor Results**
   - Check input data quality
   - Try different preprocessing steps
   - Adjust the number of components

## Contributing

We welcome contributions! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

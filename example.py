import pandas as pd
from dimension_reduction import run_analysis

# Load your data
X = pd.read_csv('your_features.csv')
y = pd.read_csv('your_target.csv')

# Run analysis
report, datasets = run_analysis(X, y, task_type='regression')

# Access optimized datasets for best methods
for method, dataset in datasets.items():
    print(f"\nOptimized dataset using {method}:")
    print(dataset.shape)
    print(dataset.head())
    
    # Save dataset
    dataset.to_csv(f'optimized_{method}.csv', index=False)
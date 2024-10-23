"""
Main execution script for dimension reduction analysis.
"""
import warnings
# Suppress all warnings
warnings.filterwarnings('ignore')
# Suppress specific scikit-learn warnings
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)

from dimension_reduction import DimensionReducer
from utils import generate_synthetic_data, save_results, save_plot
from visualization import DimensionReductionVisualizer

def run_analysis(X, y, task_type='regression', output_path='output'):
    """
    Run complete dimension reduction analysis.
    
    Parameters
    ----------
    X : pandas.DataFrame
        Input features
    y : pandas.Series
        Target variable
    task_type : str, default='regression'
        Type of task ('regression' or 'classification')
    output_path : str, default='output'
        Path to save results
    """
    # Initialize reducer and visualizer
    reducer = DimensionReducer(task_type=task_type)
    visualizer = DimensionReductionVisualizer()
    
    # Perform dimension reduction
    optimized_datasets = reducer.reduce_dimensions(X, y)
    
    # Generate visualizations
    correlation_plot = visualizer.plot_correlation_matrix(X)
    save_plot(correlation_plot, f"{task_type}_correlation_matrix", output_path)
    
    comparison_plot = visualizer.plot_reduction_comparison(reducer.results, task_type)
    save_plot(comparison_plot, f"{task_type}_reduction_comparison", output_path)
    
    # Generate and save report
    report = reducer.generate_report()
    save_results(report, optimized_datasets, output_path)
    
    return report, optimized_datasets

def main():
    """Main execution function"""
    # Run regression analysis
    print("\nREGRESSION ANALYSIS")
    print("="*50)
    
    X_reg, y_reg = generate_synthetic_data(task_type='regression')
    reg_report, reg_datasets = run_analysis(
        X_reg, y_reg,
        task_type='regression',
        output_path='regression_results'
    )
    
    print("\nRegression Analysis Results:")
    print("-" * 40)
    print("\nBest Methods:")
    for method, result in reg_report['best_methods']:
        print(f"\n{method.upper()}:")
        print(f"Score: {result['score']:.4f}")
        print(f"Number of components: {result['n_components']}")
    
    # Run classification analysis
    print("\nCLASSIFICATION ANALYSIS")
    print("="*50)
    
    X_clf, y_clf = generate_synthetic_data(task_type='classification')
    clf_report, clf_datasets = run_analysis(
        X_clf, y_clf,
        task_type='classification',
        output_path='classification_results'
    )
    
    print("\nClassification Analysis Results:")
    print("-" * 40)
    print("\nBest Methods:")
    for method, result in clf_report['best_methods']:
        print(f"\n{method.upper()}:")
        print(f"Score: {result['score']:.4f}")
        print(f"Number of components: {result['n_components']}")

if __name__ == "__main__":
    # Example usage with your own data:
    """
    # Load your data
    X = pd.read_csv('your_features.csv')
    y = pd.read_csv('your_target.csv')
    
    # For regression
    report, datasets = run_analysis(X, y, task_type='regression')
    
    # Access optimized datasets
    pca_dataset = datasets['pca']
    rf_dataset = datasets['random_forest']
    
    # Save datasets
    pca_dataset.to_csv('pca_reduced.csv', index=False)
    rf_dataset.to_csv('rf_reduced.csv', index=False)
    """
    
    main()
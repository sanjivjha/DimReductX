"""
Visualization module for dimension reduction analysis.
"""

import matplotlib.pyplot as plt
import seaborn as sns
from tabulate import tabulate
import pandas as pd
import warnings
# Suppress all warnings
warnings.filterwarnings('ignore')
# Suppress specific scikit-learn warnings
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)
class DimensionReductionVisualizer:
    """Class for creating visualizations of dimension reduction results"""
    
    def plot_correlation_matrix(self, data, title="Feature Correlation Matrix"):
        """Plot correlation matrix of features"""
        plt.figure(figsize=(12, 10))
        sns.heatmap(data.corr(), annot=True, cmap='coolwarm', center=0)
        plt.title(title)
        plt.tight_layout()
        return plt
    
    def plot_reduction_comparison(self, results, task_type='regression'):
        """Plot comparison of reduction methods"""
        plt.figure(figsize=(12, 6))
        
        for method, method_results in results.items():
            components = [r['n_components'] for r in method_results]
            scores = [r['score'] for r in method_results]
            plt.plot(components, scores, marker='o', label=method.upper())
        
        plt.xlabel('Number of Components/Features')
        plt.ylabel('R² Score' if task_type == 'regression' else 'AUC-ROC')
        plt.title('Comparison of Dimension Reduction Methods')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        return plt
    
    def plot_feature_importance(self, feature_importance, top_n=20):
        """Plot feature importance"""
        plt.figure(figsize=(12, 6))
        
        # Get top N features
        top_features = feature_importance.head(top_n)
        
        # Create bar plot
        plt.barh(range(len(top_features)), top_features['importance'])
        plt.yticks(range(len(top_features)), top_features['feature'])
        plt.xlabel('Importance Score')
        plt.title(f'Top {top_n} Most Important Features')
        plt.tight_layout()
        return plt

def create_report_tables(results, feature_importance, best_methods):
    """Create formatted tables for reporting"""
    tables = {}
    
    # Method comparison table
    method_summaries = []
    for method, method_results in results.items():
        max_result = max(method_results, key=lambda x: x['score'])
        method_summaries.append({
            'Method': method.upper(),
            'Best Score': f"{max_result['score']:.4f}",
            'Optimal Components': max_result['n_components']
        })
    
    tables['method_comparison'] = tabulate(
        method_summaries, 
        headers='keys', 
        tablefmt='pretty'
    )
    
    # Top features table
    tables['top_features'] = tabulate(
        feature_importance.head(10),
        headers='keys',
        tablefmt='pretty',
        floatfmt='.4f'
    )
    
    # Best methods summary
    best_methods_summary = pd.DataFrame(best_methods).round(4)
    tables['best_methods'] = tabulate(
        best_methods_summary,
        headers='keys',
        tablefmt='pretty',
        showindex=False
    )
    
    return tables

def format_method_results(method_name, components, score, explained_var=None):
    """Format results for display"""
    result = f"{method_name.upper()}:\n"
    result += f"Components: {components}\n"
    result += f"Score: {score:.4f}\n"
    if explained_var is not None:
        result += f"Explained Variance: {explained_var:.4f}\n"
    return result
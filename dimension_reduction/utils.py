"""
Utility functions for dimension reduction analysis.
Including file handling, data processing, and report generation.
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from fpdf import FPDF
from datetime import datetime
from tabulate import tabulate

# Suppress all warnings
warnings.filterwarnings('ignore')
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)
pd.options.mode.chained_assignment = None

class PDFReport(FPDF):
    """Custom PDF Report class with header and footer."""
    
    def header(self):
        """Add header to each page"""
        self.set_font('Arial', 'B', 15)
        self.cell(0, 10, 'Dimension Reduction Analysis Report', 0, 1, 'C')
        self.ln(5)
    
    def footer(self):
        """Add footer to each page"""
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')

def get_concise_explanation(feature_name, formula):
    """
    Generate concise feature explanation.
    
    Parameters
    ----------
    feature_name : str
        Name of the feature
    formula : str
        Mathematical formula of the feature
    
    Returns
    -------
    str
        Concise explanation of the feature
    """
    explanations = {
        'log_': "Logarithmic transform to handle exponential relationships and skewness",
        'sqrt_': "Square root transform to moderate large values and handle right-skewed data",
        'sq_': "Squared term to capture quadratic/U-shaped relationships",
        'cube_': "Cubic term to capture S-shaped curves and complex patterns",
        'prod_': "Interaction term showing multiplicative effects between features",
        'ratio_': "Relative measure between two features",
        'sum_': "Combined additive effect of features",
        'diff_': "Gap or difference between related features"
    }
    
    for prefix, explanation in explanations.items():
        if prefix in feature_name:
            return explanation
    return "Original feature from dataset"

def create_feature_importance_table(pdf, feature_importance):
    """
    Create formatted feature importance table in PDF.
    
    Parameters
    ----------
    pdf : FPDF
        PDF object
    feature_importance : pandas.DataFrame
        DataFrame containing feature importance information
    """
    # Table header
    headers = ['Rank', 'Feature', 'Importance', 'Formula', 'Stats', 'Description']
    
    # Calculate column widths
    pdf.set_font('Arial', 'B', 9)
    column_widths = [15, 35, 25, 35, 45, 75]
    
    # Header
    pdf.set_fill_color(200, 200, 200)
    y_position = pdf.get_y()
    
    for i, header in enumerate(headers):
        pdf.cell(column_widths[i], 10, header, 1, 0, 'C', True)
    pdf.ln()
    
    # Table content
    pdf.set_font('Arial', '', 8)
    for idx, row in feature_importance.head(20).iterrows():
        # Format statistics
        stats = (f"Mean: {row.get('mean', 0):.2f}\n"
                f"Corr: {row.get('correlation_with_target', 0):.2f}")
        
        # Get concise explanation
        explanation = get_concise_explanation(row['feature'], row['formula'])
        
        # Alternate row colors
        fill = idx % 2 == 0
        if fill:
            pdf.set_fill_color(240, 240, 240)
        
        # Calculate row height
        height = max(
            len(str(row['feature'])) // 15 * 5,
            len(str(row['formula'])) // 15 * 5,
            len(explanation) // 25 * 5,
            10
        )
        
        # Print row
        y_start = pdf.get_y()
        
        # Rank
        pdf.multi_cell(column_widths[0], height, str(idx + 1), 1, 'C', fill)
        pdf.set_xy(sum(column_widths[:1]) + pdf.l_margin, y_start)
        
        # Feature name
        pdf.multi_cell(column_widths[1], height, str(row['feature']), 1, 'L', fill)
        pdf.set_xy(sum(column_widths[:2]) + pdf.l_margin, y_start)
        
        # Importance score
        pdf.multi_cell(column_widths[2], height, f"{row['importance']:.4f}", 1, 'C', fill)
        pdf.set_xy(sum(column_widths[:3]) + pdf.l_margin, y_start)
        
        # Formula
        pdf.multi_cell(column_widths[3], height, str(row['formula']), 1, 'L', fill)
        pdf.set_xy(sum(column_widths[:4]) + pdf.l_margin, y_start)
        
        # Statistics
        pdf.multi_cell(column_widths[4], height, stats, 1, 'L', fill)
        pdf.set_xy(sum(column_widths[:5]) + pdf.l_margin, y_start)
        
        # Description
        pdf.multi_cell(column_widths[5], height, explanation, 1, 'L', fill)
        
        if pdf.get_y() - y_start > height:
            pdf.set_y(y_start + (pdf.get_y() - y_start))

def create_pdf_report(summary_dict, optimized_datasets, base_path="output"):
    """
    Create detailed PDF report of the analysis.
    
    Parameters
    ----------
    summary_dict : dict
        Dictionary containing analysis summary
    optimized_datasets : dict
        Dictionary containing optimized datasets
    base_path : str, default="output"
        Directory to save the report
    """
    pdf = PDFReport()
    
    # Executive Summary Page
    pdf.add_page()
    pdf.set_font('Arial', 'B', 16)
    pdf.cell(0, 10, 'Dimension Reduction Analysis Report', 0, 1, 'C')
    pdf.set_font('Arial', '', 10)
    pdf.cell(0, 10, f'Generated on: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}', 0, 1, 'C')
    
    # Analysis Overview
    pdf.ln(10)
    pdf.set_font('Arial', 'B', 14)
    pdf.cell(0, 10, '1. Analysis Overview', 0, 1)
    pdf.set_font('Arial', '', 10)
    overview_text = (
        f"Task Type: {summary_dict['task_type'].capitalize()}\n"
        f"Original Dimensions: {summary_dict['original_shape']}\n"
        f"Reduced Dimensions: {summary_dict['reduced_shape']}"
    )
    pdf.multi_cell(0, 10, overview_text)
    
    # Best Methods Summary
    pdf.ln(5)
    pdf.set_font('Arial', 'B', 14)
    pdf.cell(0, 10, '2. Best Performing Methods', 0, 1)
    pdf.set_font('Arial', '', 10)
    
    for method, result in summary_dict['best_methods']:
        method_text = (
            f"\n{method.upper()}:\n"
            f"Score: {result['score']:.4f}\n"
            f"Components: {result['n_components']}"
        )
        pdf.multi_cell(0, 10, method_text)
    
    # Feature Importance Table
    pdf.add_page()
    pdf.set_font('Arial', 'B', 14)
    pdf.cell(0, 10, '3. Feature Importance Analysis', 0, 1)
    pdf.ln(5)
    create_feature_importance_table(pdf, summary_dict['feature_importance'])
    
    # Optimized Datasets Summary
    pdf.add_page()
    pdf.set_font('Arial', 'B', 14)
    pdf.cell(0, 10, '4. Optimized Datasets', 0, 1)
    pdf.set_font('Arial', '', 10)
    
    for method, dataset in optimized_datasets.items():
        dataset_text = (
            f"\n{method.upper()}:\n"
            f"Shape: {dataset.shape}\n"
            f"Features: {', '.join(dataset.columns)}"
        )
        pdf.multi_cell(0, 10, dataset_text)
    
    # Save PDF
    path = create_output_path(base_path)
    pdf_path = f"{path}/analysis_report.pdf"
    pdf.output(pdf_path)
    print(f"PDF report saved to: {pdf_path}")

def create_output_path(base_path="output"):
    """Create output directory if it doesn't exist"""
    if not os.path.exists(base_path):
        os.makedirs(base_path)
    return base_path

def save_plot(plt, name, base_path="output"):
    """Save plot to output directory"""
    path = create_output_path(base_path)
    plt.savefig(f"{path}/{name}.png", bbox_inches='tight', dpi=300)
    plt.close()

def save_results(summary_dict, optimized_datasets, base_path="output"):
    """
    Save analysis results to files.
    
    Parameters
    ----------
    summary_dict : dict
        Dictionary containing analysis summary
    optimized_datasets : dict
        Dictionary containing optimized datasets
    base_path : str, default="output"
        Directory to save results
    """
    path = create_output_path(base_path)
    
    # Create PDF report
    create_pdf_report(summary_dict, optimized_datasets, base_path)
    
    # Save summary as text
    with open(f"{path}/analysis_summary.txt", "w") as f:
        f.write("DIMENSION REDUCTION ANALYSIS SUMMARY\n")
        f.write("="*40 + "\n\n")
        for key, value in summary_dict.items():
            if isinstance(value, pd.DataFrame):
                f.write(f"\n{key}:\n{value.to_string()}\n")
            else:
                f.write(f"{key}: {value}\n")
    
    # Save optimized datasets
    for method, data in optimized_datasets.items():
        data.to_csv(f"{path}/optimized_dataset_{method}.csv", index=False)
    
    print(f"\nResults saved to {path}/")

def generate_synthetic_data(n_samples=2500, n_features=22, task_type='regression'):
    """
    Generate synthetic data for testing.
    
    Parameters
    ----------
    n_samples : int, default=2500
        Number of samples to generate
    n_features : int, default=22
        Number of features to generate
    task_type : str, default='regression'
        Type of task ('regression' or 'classification')
    
    Returns
    -------
    tuple
        (X, y) pair of features and target
    """
    np.random.seed(42)
    
    # Create base features
    base_features = np.random.randn(n_samples, 5)
    noise = np.random.randn(n_samples, n_features) * 0.1
    
    data = np.zeros((n_samples, n_features))
    data[:, 0:5] = base_features
    data[:, 5:10] = base_features + noise[:, 5:10]
    data[:, 10:15] = base_features * 1.5 + noise[:, 10:15]
    data[:, 15:20] = base_features * 0.5 + noise[:, 15:20]
    data[:, 20:] = base_features[:, :2] * 2 + noise[:, 20:]
    
    # Create target with non-linear relationships
    if task_type == 'regression':
        target = (2 * data[:, 0]**2 + np.log(np.abs(data[:, 1]) + 1) - 
                 np.exp(data[:, 2] * 0.1) + data[:, 3] * data[:, 4] +
                 np.random.randn(n_samples) * 0.1)
    else:
        target = (2 * data[:, 0]**2 + data[:, 1] * data[:, 2] - 
                 np.exp(data[:, 3] * 0.1) > 0).astype(int)
    
    X = pd.DataFrame(data, columns=[f'feature_{i+1}' for i in range(n_features)])
    y = pd.Series(target, name='target')
    
    return X, y
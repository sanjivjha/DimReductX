"""
Feature discovery and generation module.
Handles the creation and selection of derived features.
"""

import numpy as np
import pandas as pd
from itertools import combinations
from sklearn.preprocessing import PolynomialFeatures
import warnings
# Suppress all warnings
warnings.filterwarnings('ignore')
# Suppress specific scikit-learn warnings
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)
class FeatureDiscovery:
    """Class to handle feature generation and selection"""
    
    def __init__(self, max_poly_degree=2, max_combinations=2):
        self.max_poly_degree = max_poly_degree
        self.max_combinations = max_combinations
        self.feature_formulas = {}
    
    def generate_polynomial_features(self, X):
        """Generate polynomial features"""
        poly = PolynomialFeatures(degree=self.max_poly_degree, include_bias=False)
        poly_features = poly.fit_transform(X)
        feature_names = poly.get_feature_names_out(X.columns)
        return pd.DataFrame(poly_features, columns=feature_names, index=X.index)
    
    def generate_mathematical_features(self, X):
        """Generate features using mathematical transformations"""
        new_features = pd.DataFrame(index=X.index)
        orig_cols = X.columns
        
        # Single column transformations
        for col in orig_cols:
            if X[col].min() > 0:
                new_features[f'log_{col}'] = np.log(X[col])
                self.feature_formulas[f'log_{col}'] = f'log({col})'
                new_features[f'sqrt_{col}'] = np.sqrt(X[col])
                self.feature_formulas[f'sqrt_{col}'] = f'sqrt({col})'
            
            new_features[f'sq_{col}'] = X[col] ** 2
            new_features[f'cube_{col}'] = X[col] ** 3
            self.feature_formulas[f'sq_{col}'] = f'({col})²'
            self.feature_formulas[f'cube_{col}'] = f'({col})³'
        
        # Interactions
        for col1, col2 in combinations(orig_cols, 2):
            new_features[f'prod_{col1}_{col2}'] = X[col1] * X[col2]
            self.feature_formulas[f'prod_{col1}_{col2}'] = f'{col1} × {col2}'
            
            if (X[col2] != 0).all():
                new_features[f'ratio_{col1}_{col2}'] = X[col1] / X[col2]
                self.feature_formulas[f'ratio_{col1}_{col2}'] = f'{col1} ÷ {col2}'
            
            new_features[f'sum_{col1}_{col2}'] = X[col1] + X[col2]
            new_features[f'diff_{col1}_{col2}'] = X[col1] - X[col2]
            self.feature_formulas[f'sum_{col1}_{col2}'] = f'{col1} + {col2}'
            self.feature_formulas[f'diff_{col1}_{col2}'] = f'{col1} - {col2}'
        
        return new_features
    
    def generate_all_features(self, X):
        """Generate all possible derived features"""
        print("Generating polynomial features...")
        poly_features = self.generate_polynomial_features(X)
        
        print("Generating mathematical features...")
        math_features = self.generate_mathematical_features(X)
        
        all_features = pd.concat([X, poly_features, math_features], axis=1)
        print("Removing highly correlated features...")
        all_features = self._remove_highly_correlated(all_features)
        
        return all_features
    
    def _remove_highly_correlated(self, df, threshold=0.95):
        """Remove highly correlated features"""
        corr_matrix = df.corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
        return df.drop(columns=to_drop)
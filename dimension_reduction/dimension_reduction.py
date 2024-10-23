"""
Comprehensive dimension reduction implementation.

This module provides the main functionality for:
- Dimension reduction using multiple methods
- Feature importance analysis
- Performance evaluation
- Results generation
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, f_regression, f_classif
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression, LassoCV
from sklearn.metrics import r2_score, roc_auc_score
from feature_discovery import FeatureDiscovery
from visualization import DimensionReductionVisualizer, create_report_tables
import warnings
# Suppress all warnings
warnings.filterwarnings('ignore')
# Suppress specific scikit-learn warnings
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)

class DimensionReducer:
    """
    Class for performing dimension reduction using multiple methods.
    
    Parameters
    ----------
    task_type : str, default='regression'
        Type of task ('regression' or 'classification')
    max_poly_degree : int, default=2
        Maximum degree for polynomial feature generation
        
    Attributes
    ----------
    scaler : StandardScaler
        Scaler for feature standardization
    feature_discovery : FeatureDiscovery
        Instance of FeatureDiscovery class
    visualizer : DimensionReductionVisualizer
        Instance of visualization class
    results : dict
        Dictionary storing results of different methods
    """
    
    def __init__(self, task_type='regression', max_poly_degree=2):
        self.task_type = task_type
        self.scaler = StandardScaler()
        self.feature_discovery = FeatureDiscovery(max_poly_degree=max_poly_degree)
        self.visualizer = DimensionReductionVisualizer()
        self.results = {}
        self.best_methods = []
        
    def _prepare_data(self, X, y):
        """
        Prepare data for analysis.
        
        Parameters
        ----------
        X : pandas.DataFrame
            Input features
        y : pandas.Series
            Target variable
        """
        # Store original data
        self.X_original = X.copy()
        self.y = y
        
        # Generate derived features
        print("\nGenerating derived features...")
        self.X = self.feature_discovery.generate_all_features(X)
        self.feature_names = self.X.columns
        
        # Split data
        print("Splitting and scaling data...")
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42
        )
        
        # Scale features
        self.X_scaled = pd.DataFrame(
            self.scaler.fit_transform(self.X),
            columns=self.feature_names
        )
        self.X_train_scaled = pd.DataFrame(
            self.scaler.transform(self.X_train),
            columns=self.feature_names
        )
        self.X_test_scaled = pd.DataFrame(
            self.scaler.transform(self.X_test),
            columns=self.feature_names
        )
        
    def reduce_dimensions(self, X, y, methods=None, n_components_list=None):
        """
        Perform dimension reduction using specified methods.
        
        Parameters
        ----------
        X : pandas.DataFrame
            Input features
        y : pandas.Series
            Target variable
        methods : list, optional
            List of methods to use ('pca', 'select_k_best', 'random_forest')
        n_components_list : list, optional
            List of number of components to try
            
        Returns
        -------
        dict
            Dictionary containing optimized datasets
        """
        # Set defaults
        if methods is None:
            methods = ['pca', 'select_k_best', 'random_forest']
        if n_components_list is None:
            n_components_list = [5, 10, 15]
            
        # Prepare data
        self._prepare_data(X, y)
        
        # Run each method
        for method in methods:
            print(f"\nApplying {method.upper()} method...")
            self.results[method] = []
            
            for n_components in n_components_list:
                print(f"Testing with {n_components} components...")
                result = self._apply_reduction_method(method, n_components)
                self.results[method].append(result)
        
        # Find best methods
        self._find_best_methods()
        
        return self.get_optimized_datasets()
        
    def _apply_reduction_method(self, method, n_components):
        """
        Apply specific reduction method.
        
        Parameters
        ----------
        method : str
            Reduction method to use
        n_components : int
            Number of components
            
        Returns
        -------
        dict
            Results of the reduction method
        """
        if method == 'pca':
            return self._reduce_pca(n_components)
        elif method == 'select_k_best':
            return self._reduce_select_k_best(n_components)
        else:  # random_forest
            return self._reduce_random_forest(n_components)
            
    def _reduce_pca(self, n_components):
        """Apply PCA reduction"""
        pca = PCA(n_components=n_components)
        X_train_reduced = pca.fit_transform(self.X_train_scaled)
        X_test_reduced = pca.transform(self.X_test_scaled)
        
        score = self._evaluate_reduction(X_train_reduced, X_test_reduced)
        
        return {
            'n_components': n_components,
            'score': score,
            'explained_variance': np.sum(pca.explained_variance_ratio_),
            'reducer': pca
        }
        
    def _reduce_select_k_best(self, n_components):
        """Apply SelectKBest reduction"""
        selector = SelectKBest(
            score_func=f_regression if self.task_type == 'regression' else f_classif,
            k=n_components
        )
        X_train_reduced = selector.fit_transform(self.X_train_scaled, self.y_train)
        X_test_reduced = selector.transform(self.X_test_scaled)
        
        score = self._evaluate_reduction(X_train_reduced, X_test_reduced)
        
        return {
            'n_components': n_components,
            'score': score,
            'reducer': selector,
            'selected_features': self.X_scaled.columns[selector.get_support()].tolist()
        }
        
    def _reduce_random_forest(self, n_components):
        """Apply Random Forest based feature selection"""
        model = (RandomForestRegressor(n_estimators=100, random_state=42)
                if self.task_type == 'regression'
                else RandomForestClassifier(n_estimators=100, random_state=42))
        
        model.fit(self.X_train_scaled, self.y_train)
        importance = pd.Series(model.feature_importances_, index=self.feature_names)
        top_features = importance.nlargest(n_components).index
        
        X_train_reduced = self.X_train_scaled[top_features]
        X_test_reduced = self.X_test_scaled[top_features]
        
        score = self._evaluate_reduction(X_train_reduced, X_test_reduced)
        
        return {
            'n_components': n_components,
            'score': score,
            'top_features': top_features,
            'importance': importance
        }
        
    def _evaluate_reduction(self, X_train_reduced, X_test_reduced):
        """
        Evaluate reduction performance.
        
        Parameters
        ----------
        X_train_reduced : array-like
            Reduced training features
        X_test_reduced : array-like
            Reduced test features
            
        Returns
        -------
        float
            Performance score
        """
        if self.task_type == 'regression':
            model = LinearRegression()
        else:
            model = LogisticRegression(random_state=42)
        
        model.fit(X_train_reduced, self.y_train)
        
        if self.task_type == 'regression':
            y_pred = model.predict(X_test_reduced)
            return r2_score(self.y_test, y_pred)
        else:
            y_pred_proba = model.predict_proba(X_test_reduced)[:, 1]
            return roc_auc_score(self.y_test, y_pred_proba)
            
    def _find_best_methods(self):
        """Find the two best performing methods."""
        method_scores = []
        for method, results in self.results.items():
            best_score = max(results, key=lambda x: x['score'])
            method_scores.append((method, best_score))
        
        # Sort by score descending
        method_scores.sort(key=lambda x: x[1]['score'], reverse=True)
        self.best_methods = method_scores[:2]
        
    def get_optimized_datasets(self):
        """
        Get optimized datasets for the best methods.
        
        Returns
        -------
        dict
            Dictionary containing reduced datasets
        """
        optimized_datasets = {}
        
        for method, best_result in self.best_methods:
            if method == 'pca':
                reduced_data = best_result['reducer'].transform(self.X_scaled)
                columns = [f'PC_{i+1}' for i in range(best_result['n_components'])]
            elif method == 'select_k_best':
                reduced_data = best_result['reducer'].transform(self.X_scaled)
                columns = best_result['selected_features']
            else:  # random_forest
                reduced_data = self.X_scaled[best_result['top_features']]
                columns = best_result['top_features'].tolist()
            
            optimized_datasets[method] = pd.DataFrame(reduced_data, columns=columns)
        
        return optimized_datasets
    
    def _get_feature_statistics(self, feature):
        """
        Calculate additional statistics for a feature.
        
        Parameters
        ----------
        feature : str
            Feature name
            
        Returns
        -------
        dict
            Dictionary containing feature statistics
        """
        stats = {
            'mean': self.X[feature].mean(),
            'std': self.X[feature].std(),
            'min': self.X[feature].min(),
            'max': self.X[feature].max(),
            'correlation_with_target': np.corrcoef(self.X[feature], self.y)[0,1]
        }
        return stats
        
    def _get_feature_importance(self):
        """
        Calculate feature importance with additional information.
        
        Returns
        -------
        pandas.DataFrame
            DataFrame containing feature importance information
        """
        if self.task_type == 'regression':
            model = LassoCV(cv=5)
        else:
            model = LogisticRegression(penalty='l1', solver='liblinear', random_state=42)
        
        model.fit(self.X_scaled, self.y)
        importance = np.abs(model.coef_) if self.task_type == 'regression' else np.abs(model.coef_[0])
        
        # Create base DataFrame
        feature_importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importance,
            'formula': [self.feature_discovery.feature_formulas.get(f, f) 
                       for f in self.feature_names]
        })
        
        # Add statistics for each feature
        stats_df = pd.DataFrame([
            self._get_feature_statistics(feature)
            for feature in self.feature_names
        ])
        
        feature_importance = pd.concat([feature_importance, stats_df], axis=1)
        return feature_importance.sort_values('importance', ascending=False)
        
    def generate_report(self):
        """
        Generate comprehensive analysis report.
        
        Returns
        -------
        dict
            Dictionary containing analysis summary
        """
        feature_importance = self._get_feature_importance()
        
        summary = {
            'task_type': self.task_type,
            'original_shape': self.X.shape,
            'reduced_shape': (self.X.shape[0], self.best_methods[0][1]['n_components']),
            'best_methods': self.best_methods,
            'feature_importance': feature_importance
        }
        
        return summary
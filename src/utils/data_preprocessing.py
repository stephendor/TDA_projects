"""
Data preprocessing utilities for TDA applications.
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Optional, Union, Dict, Any
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler


def time_delay_embedding(
    time_series: np.ndarray, 
    embedding_dim: int, 
    delay: int = 1
) -> np.ndarray:
    """
    Create time delay embedding of a time series.
    
    Parameters:
    -----------
    time_series : np.ndarray
        1D time series data
    embedding_dim : int
        Embedding dimension
    delay : int
        Time delay
        
    Returns:
    --------
    np.ndarray
        Embedded time series of shape (n_points, embedding_dim)
    """
    if len(time_series) < embedding_dim * delay:
        raise ValueError("Time series too short for given embedding parameters")
    
    # Create delay matrix
    n_points = len(time_series) - (embedding_dim - 1) * delay
    embedded = np.zeros((n_points, embedding_dim))
    
    for i in range(embedding_dim):
        start_idx = i * delay
        end_idx = start_idx + n_points
        embedded[:, i] = time_series[start_idx:end_idx]
    
    return embedded


def sliding_window_embedding(
    data: np.ndarray,
    window_size: int,
    stride: int = 1
) -> np.ndarray:
    """
    Create sliding window embedding of data.
    
    Parameters:
    -----------
    data : np.ndarray
        Input data
    window_size : int
        Size of sliding window
    stride : int
        Stride between windows
        
    Returns:
    --------
    np.ndarray
        Windowed data
    """
    if len(data) < window_size:
        return data.reshape(1, -1)
    
    n_windows = (len(data) - window_size) // stride + 1
    windowed = np.zeros((n_windows, window_size))
    
    for i in range(n_windows):
        start_idx = i * stride
        windowed[i] = data[start_idx:start_idx + window_size]
    
    return windowed


def normalize_data(
    data: np.ndarray,
    method: str = 'standard',
    fit_data: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, Any]:
    """
    Normalize data using various scaling methods.
    
    Parameters:
    -----------
    data : np.ndarray
        Data to normalize
    method : str
        Normalization method ('standard', 'minmax', 'robust')
    fit_data : np.ndarray, optional
        Data to fit scaler on (if different from data)
        
    Returns:
    --------
    Tuple[np.ndarray, Any]
        Normalized data and fitted scaler
    """
    if method == 'standard':
        scaler = StandardScaler()
    elif method == 'minmax':
        scaler = MinMaxScaler()
    elif method == 'robust':
        scaler = RobustScaler()
    else:
        raise ValueError(f"Unknown normalization method: {method}")
    
    fit_data = fit_data if fit_data is not None else data
    scaler.fit(fit_data.reshape(-1, 1) if len(fit_data.shape) == 1 else fit_data)
    
    normalized = scaler.transform(data.reshape(-1, 1) if len(data.shape) == 1 else data)
    
    return normalized, scaler


def remove_outliers(
    data: np.ndarray,
    method: str = 'iqr',
    threshold: float = 1.5
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Remove outliers from data.
    
    Parameters:
    -----------
    data : np.ndarray
        Input data
    method : str
        Outlier detection method ('iqr', 'zscore')
    threshold : float
        Threshold for outlier detection
        
    Returns:
    --------
    Tuple[np.ndarray, np.ndarray]
        Cleaned data and outlier mask
    """
    if method == 'iqr':
        Q1 = np.percentile(data, 25)
        Q3 = np.percentile(data, 75)
        IQR = Q3 - Q1
        lower_bound = Q1 - threshold * IQR
        upper_bound = Q3 + threshold * IQR
        outlier_mask = (data < lower_bound) | (data > upper_bound)
    elif method == 'zscore':
        z_scores = np.abs((data - np.mean(data)) / np.std(data))
        outlier_mask = z_scores > threshold
    else:
        raise ValueError(f"Unknown outlier detection method: {method}")
    
    cleaned_data = data[~outlier_mask]
    return cleaned_data, outlier_mask


def interpolate_missing_values(
    data: np.ndarray,
    method: str = 'linear'
) -> np.ndarray:
    """
    Interpolate missing values in time series data.
    
    Parameters:
    -----------
    data : np.ndarray
        Data with missing values (NaN)
    method : str
        Interpolation method
        
    Returns:
    --------
    np.ndarray
        Data with interpolated values
    """
    if not np.any(np.isnan(data)):
        return data
    
    # Simple linear interpolation
    valid_indices = ~np.isnan(data)
    if not np.any(valid_indices):
        return data  # All NaN, can't interpolate
    
    interpolated = data.copy()
    if method == 'linear':
        # Forward fill then backward fill
        valid_data = data[valid_indices]
        valid_idx = np.where(valid_indices)[0]
        
        for i in range(len(data)):
            if np.isnan(data[i]):
                # Find nearest valid values
                lower_idx = valid_idx[valid_idx < i]
                upper_idx = valid_idx[valid_idx > i]
                
                if len(lower_idx) > 0 and len(upper_idx) > 0:
                    # Linear interpolation
                    x1, y1 = lower_idx[-1], data[lower_idx[-1]]
                    x2, y2 = upper_idx[0], data[upper_idx[0]]
                    interpolated[i] = y1 + (y2 - y1) * (i - x1) / (x2 - x1)
                elif len(lower_idx) > 0:
                    # Forward fill
                    interpolated[i] = data[lower_idx[-1]]
                elif len(upper_idx) > 0:
                    # Backward fill
                    interpolated[i] = data[upper_idx[0]]
    
    return interpolated


class DataPreprocessor:
    """
    Comprehensive data preprocessing for TDA applications.
    """
    
    def __init__(
        self,
        normalize: bool = True,
        normalization_method: str = 'standard',
        remove_outliers: bool = False,
        outlier_method: str = 'iqr',
        outlier_threshold: float = 1.5,
        interpolate_missing: bool = True,
        interpolation_method: str = 'linear'
    ):
        """
        Initialize data preprocessor.
        
        Parameters:
        -----------
        normalize : bool
            Whether to normalize data
        normalization_method : str
            Method for normalization
        remove_outliers : bool
            Whether to remove outliers
        outlier_method : str
            Method for outlier detection
        outlier_threshold : float
            Threshold for outlier detection
        interpolate_missing : bool
            Whether to interpolate missing values
        interpolation_method : str
            Method for interpolation
        """
        self.normalize = normalize
        self.normalization_method = normalization_method
        self.remove_outliers = remove_outliers
        self.outlier_method = outlier_method
        self.outlier_threshold = outlier_threshold
        self.interpolate_missing = interpolate_missing
        self.interpolation_method = interpolation_method
        
        self.scaler = None
        self.is_fitted = False
    
    def fit(self, data: np.ndarray) -> 'DataPreprocessor':
        """
        Fit preprocessor to data.
        
        Parameters:
        -----------
        data : np.ndarray
            Training data
            
        Returns:
        --------
        self : DataPreprocessor
        """
        processed_data = data.copy()
        
        # Handle missing values
        if self.interpolate_missing:
            processed_data = interpolate_missing_values(
                processed_data, 
                self.interpolation_method
            )
        
        # Remove outliers
        if self.remove_outliers:
            processed_data, _ = remove_outliers(
                processed_data,
                self.outlier_method,
                self.outlier_threshold
            )
        
        # Fit normalizer
        if self.normalize:
            _, self.scaler = normalize_data(
                processed_data,
                self.normalization_method
            )
        
        self.is_fitted = True
        return self
    
    def transform(self, data: np.ndarray) -> np.ndarray:
        """
        Transform data using fitted preprocessor.
        
        Parameters:
        -----------
        data : np.ndarray
            Data to transform
            
        Returns:
        --------
        np.ndarray
            Transformed data
        """
        if not self.is_fitted:
            raise ValueError("Preprocessor must be fitted before transform")
        
        processed_data = data.copy()
        
        # Handle missing values
        if self.interpolate_missing:
            processed_data = interpolate_missing_values(
                processed_data,
                self.interpolation_method
            )
        
        # Remove outliers
        if self.remove_outliers:
            processed_data, _ = remove_outliers(
                processed_data,
                self.outlier_method,
                self.outlier_threshold
            )
        
        # Normalize
        if self.normalize and self.scaler is not None:
            shape = processed_data.shape
            if len(shape) == 1:
                processed_data = self.scaler.transform(processed_data.reshape(-1, 1)).flatten()
            else:
                processed_data = self.scaler.transform(processed_data)
        
        return processed_data
    
    def fit_transform(self, data: np.ndarray) -> np.ndarray:
        """
        Fit preprocessor and transform data.
        
        Parameters:
        -----------
        data : np.ndarray
            Data to fit and transform
            
        Returns:
        --------
        np.ndarray
            Transformed data
        """
        return self.fit(data).transform(data)


def preprocess_data(data: np.ndarray, **kwargs) -> np.ndarray:
    """
    Convenience function for data preprocessing.
    
    Parameters:
    -----------
    data : np.ndarray
        Data to preprocess
    **kwargs
        Additional parameters for DataPreprocessor
        
    Returns:
    --------
    np.ndarray
        Preprocessed data
    """
    preprocessor = DataPreprocessor(**kwargs)
    return preprocessor.fit_transform(data)

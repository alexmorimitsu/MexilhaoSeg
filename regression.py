#!/usr/bin/env python3
"""
Regression script that loads a trained neural network model and applies it to statistics.csv

This script:
1. Loads the trained regression model from optimized_regression_model.pth
2. Reads the statistics.csv file
3. Applies the same preprocessing (feature selection and normalization)
4. Runs inference on the data
5. Adds regression results as a new column to the CSV
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import argparse
import os
from pathlib import Path


class SimpleNeuralNetwork(nn.Module):
    """
    Simplified neural network optimized for small datasets
    (Same architecture as used in training)
    """
    def __init__(self, input_size, hidden_size=16, output_size=1, dropout_rate=0.1):
        super(SimpleNeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(dropout_rate)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Apply trained regression model to statistics.csv'
    )
    
    parser.add_argument('--model_path', type=str, 
                       default='./Modelos/regression_network.pth',
                       help='Path to trained model file (default: ./Modelos/regression_network.pth)')
    parser.add_argument('--statistics_csv', type=str, 
                       default='./outputs/statistics.csv',
                       help='Path to statistics CSV file (default: ./outputs/statistics.csv)')
    parser.add_argument('--output_csv', type=str, 
                       default='./outputs/resultados.csv',
                       help='Path to output CSV file (default: ./outputs/resultados.csv)')
    
    return parser.parse_args()


def load_trained_model(model_path):
    """
    Load the trained model and return model, feature columns, and normalization parameters
    
    Args:
        model_path: Path to the .pth model file
        
    Returns:
        Tuple of (model, feature_columns, mean, std, loss_function_name)
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    print(f"Loading model from: {model_path}")
    
    # Load the checkpoint
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    
    # Extract information from checkpoint
    feature_columns = checkpoint['feature_columns']
    loss_function_name = checkpoint.get('loss_function_name', 'Unknown')
    
    # Get normalization parameters (if saved)
    mean = checkpoint.get('mean', None)
    std = checkpoint.get('std', None)
    
    # Create model with correct architecture
    input_size = len(feature_columns)
    model = SimpleNeuralNetwork(input_size=input_size, hidden_size=16, dropout_rate=0.1)
    
    # Load model state
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()  # Set to evaluation mode
    
    print(f"Model loaded successfully!")
    print(f"  - Loss function used: {loss_function_name}")
    print(f"  - Feature columns: {feature_columns}")
    print(f"  - Input size: {input_size}")
    
    return model, feature_columns, mean, std, loss_function_name


def normalize_features(X, mean, std):
    """
    Apply normalization to features using saved mean and std
    
    Args:
        X: Feature matrix
        mean: Mean values for normalization
        std: Standard deviation values for normalization
        
    Returns:
        Normalized feature matrix
    """
    if mean is None or std is None:
        print("Warning: No normalization parameters found in model. Using current data statistics.")
        mean = np.mean(X, axis=0)
        std = np.std(X, axis=0)
        std = np.where(std == 0, 1, std)  # Avoid division by zero
    
    X_norm = (X - mean) / std
    return X_norm


def apply_regression_to_csv(model_path, statistics_csv, output_csv):
    """
    Apply regression model to statistics CSV and save results
    
    Args:
        model_path: Path to trained model
        statistics_csv: Path to input statistics CSV
        output_csv: Path to output CSV with regression results
    """
    # Load the trained model
    model, feature_columns, mean, std, loss_function_name = load_trained_model(model_path)
    
    # Read statistics CSV
    if not os.path.exists(statistics_csv):
        raise FileNotFoundError(f"Statistics CSV not found: {statistics_csv}")
    
    print(f"\nReading statistics from: {statistics_csv}")
    df = pd.read_csv(statistics_csv)
    print(f"Loaded {len(df)} rows")
    
    # Check if required feature columns exist
    missing_features = [col for col in feature_columns if col not in df.columns]
    if missing_features:
        raise ValueError(f"Missing required features in CSV: {missing_features}")
    
    # Extract features
    X = df[feature_columns].values
    print(f"Extracted features shape: {X.shape}")
    
    # Apply normalization
    X_normalized = normalize_features(X, mean, std)
    
    # Convert to PyTorch tensor
    X_tensor = torch.FloatTensor(X_normalized)
    
    # Run inference
    print("\nRunning inference...")
    with torch.no_grad():
        predictions = model(X_tensor)
        predictions_np = predictions.numpy().flatten()
    
    # Create new dataframe with specified columns
    result_df = pd.DataFrame({
        'Imagem': df['filename'].str.replace('_mask_labels.npy', '', regex=False),
        'Área de cobertura': df['coverage_percentage'],
        'Número de mexilhões (previsto)': predictions_np
    })
    
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(output_csv)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # Save results
    result_df.to_csv(output_csv, index=False)
    
    print(f"\nResults saved to: {output_csv}")
    print(f"Added regression predictions for {len(predictions_np)} samples")
    print(f"Prediction range: {predictions_np.min():.3f} to {predictions_np.max():.3f}")
    print(f"Mean prediction: {predictions_np.mean():.3f}")
    
    return result_df


def main():
    """Main function."""
    args = parse_arguments()
    
    try:
        # Apply regression to CSV
        result_df = apply_regression_to_csv(
            args.model_path,
            args.statistics_csv,
            args.output_csv
        )
        
        print("\n✅ Regression analysis completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit_code = main()
    exit(exit_code)

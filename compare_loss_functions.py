import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

class SimpleNeuralNetwork(nn.Module):
    """
    Simplified neural network optimized for small datasets
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

class MAELoss(nn.Module):
    """
    Mean Absolute Error Loss
    """
    def __init__(self):
        super(MAELoss, self).__init__()
    
    def forward(self, pred, true):
        return torch.mean(torch.abs(pred - true))

class HuberLoss(nn.Module):
    """
    Huber Loss - robust to outliers
    """
    def __init__(self, delta=1.0):
        super(HuberLoss, self).__init__()
        self.delta = delta
    
    def forward(self, pred, true):
        error = pred - true
        abs_error = torch.abs(error)
        
        huber_loss = torch.where(
            abs_error <= self.delta,
            0.5 * error ** 2,
            self.delta * abs_error - 0.5 * self.delta ** 2
        )
        
        return torch.mean(huber_loss)

class SmoothL1Loss(nn.Module):
    """
    Smooth L1 Loss (same as Huber with delta=1)
    """
    def __init__(self):
        super(SmoothL1Loss, self).__init__()
    
    def forward(self, pred, true):
        error = pred - true
        abs_error = torch.abs(error)
        
        smooth_l1 = torch.where(
            abs_error <= 1.0,
            0.5 * error ** 2,
            abs_error - 0.5
        )
        
        return torch.mean(smooth_l1)

class FocalLoss(nn.Module):
    """
    Focal Loss for regression - focuses on hard examples
    """
    def __init__(self, alpha=2.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
    
    def forward(self, pred, true):
        error = torch.abs(pred - true)
        focal_weight = torch.pow(error, self.alpha)
        return torch.mean(focal_weight * error)

class QuantileLoss(nn.Module):
    """
    Quantile Loss for robust regression
    """
    def __init__(self, quantile=0.5):
        super(QuantileLoss, self).__init__()
        self.quantile = quantile
    
    def forward(self, pred, true):
        error = true - pred
        loss = torch.max(
            (self.quantile - 1) * error,
            self.quantile * error
        )
        return torch.mean(loss)

def normalize_features(X_train, X_val):
    """
    Robust feature normalization
    """
    mean = np.mean(X_train, axis=0)
    std = np.std(X_train, axis=0)
    std = np.where(std == 0, 1, std)
    
    X_train_norm = (X_train - mean) / std
    X_val_norm = (X_val - mean) / std
    
    return X_train_norm, X_val_norm, mean, std

def calculate_metrics(y_true, y_pred):
    """
    Calculate regression metrics
    """
    mse = np.mean((y_true - y_pred) ** 2)
    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(mse)
    
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
    
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    max_error = np.max(np.abs(y_true - y_pred))
    
    return {
        'mse': mse,
        'mae': mae,
        'rmse': rmse,
        'r2': r2,
        'mape': mape,
        'max_error': max_error
    }

def load_and_preprocess_data(train_path, val_path):
    """
    Load and preprocess data with top features
    """
    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    
    # Use top 5 most correlated features
    top_features = [
        'num_connected_components',
        'coverage_percentage', 
        'dilated_max_area',
        'dilated_std_area',
        'dilated_avg_area'
    ]
    
    all_features = [col for col in train_df.columns if col not in ['filename', 'anotacao']]
    feature_columns = [f for f in top_features if f in all_features]
    
    print(f"Using {len(feature_columns)} features: {feature_columns}")
    
    # Extract features and targets
    X_train = train_df[feature_columns].values
    y_train = train_df['anotacao'].values
    X_val = val_df[feature_columns].values
    y_val = val_df['anotacao'].values
    
    # Normalize features
    X_train_scaled, X_val_scaled, mean, std = normalize_features(X_train, X_val)
    
    # Convert to PyTorch tensors
    X_train_tensor = torch.FloatTensor(X_train_scaled)
    y_train_tensor = torch.FloatTensor(y_train).reshape(-1, 1)
    X_val_tensor = torch.FloatTensor(X_val_scaled)
    y_val_tensor = torch.FloatTensor(y_val).reshape(-1, 1)
    
    return X_train_tensor, y_train_tensor, X_val_tensor, y_val_tensor, val_df, feature_columns

def train_model_with_loss(model, X_train, y_train, X_val, y_val, loss_function, loss_name, epochs=150, learning_rate=0.01, batch_size=4, patience=15):
    """
    Train model with specific loss function
    """
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.01)
    
    train_losses = []
    val_losses = []
    
    best_val_loss = float('inf')
    best_model_state = None
    best_epoch = 0
    patience_counter = 0
    
    print(f"Training with {loss_name}...")
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = loss_function(outputs, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()
        
        # Validation phase
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val)
            val_loss = loss_function(val_outputs, y_val).item()
        
        train_losses.append(train_loss / len(train_loader))
        val_losses.append(val_loss)
        
        # Early stopping
        if val_loss < best_val_loss and not torch.isinf(torch.tensor(val_loss)) and not torch.isnan(torch.tensor(val_loss)):
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()
            best_epoch = epoch
            patience_counter = 0
        else:
            patience_counter += 1
        
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch+1}, best val loss: {best_val_loss:.4f}")
            break
    
    # Load best weights (only if we have a valid state)
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    else:
        print(f"Warning: No valid model state found for {loss_name}, using final weights")
    
    return train_losses, val_losses, best_epoch, best_val_loss

def evaluate_model(model, X_val, y_val, loss_function):
    """
    Evaluate model and return predictions and metrics
    """
    model.eval()
    with torch.no_grad():
        predictions = model(X_val)
        predictions_np = predictions.numpy().flatten()
        y_val_np = y_val.numpy().flatten()
    
    metrics = calculate_metrics(y_val_np, predictions_np)
    loss_value = loss_function(predictions, y_val).item()
    metrics['loss_value'] = loss_value
    
    return metrics, predictions_np, y_val_np

def compare_loss_functions():
    """
    Compare different loss functions and save all predictions in one CSV
    """
    # Set random seeds
    torch.manual_seed(42)
    np.random.seed(42)
    
    print("🚀 Comparing Loss Functions")
    print("="*60)
    
    # Load data
    X_train, y_train, X_val, y_val, val_df, feature_columns = load_and_preprocess_data(
        'Mexilhoes_dados_treino.csv', 
        'Mexilhoes_dados_validacao.csv'
    )
    
    print(f"Training set shape: {X_train.shape}")
    print(f"Validation set shape: {X_val.shape}")
    
    # Define loss functions
    loss_functions = {
        'MSE': nn.MSELoss(),
        'MAE': MAELoss(),
        'Huber': HuberLoss(delta=1.0),
        'Smooth_L1': SmoothL1Loss(),
        'Focal': FocalLoss(alpha=2.0),
        'Quantile': QuantileLoss(quantile=0.5)
    }
    
    # Initialize results storage
    all_predictions = {}
    all_metrics = {}
    training_info = {}
    
    # Train and evaluate each loss function
    for loss_name, loss_function in loss_functions.items():
        print(f"\n{'='*50}")
        print(f"Training with {loss_name} Loss")
        print(f"{'='*50}")
        
        # Create fresh model for each loss function
        input_size = X_train.shape[1]
        model = SimpleNeuralNetwork(input_size=input_size, hidden_size=16, dropout_rate=0.1)
        
        # Train model
        train_losses, val_losses, best_epoch, best_val_loss = train_model_with_loss(
            model, X_train, y_train, X_val, y_val, loss_function, loss_name
        )
        
        # Evaluate model
        metrics, predictions, actual = evaluate_model(model, X_val, y_val, loss_function)
        
        # Store results
        all_predictions[loss_name] = predictions
        all_metrics[loss_name] = metrics
        training_info[loss_name] = {
            'best_epoch': best_epoch,
            'best_val_loss': best_val_loss,
            'total_epochs': len(train_losses)
        }
        
        # Print results
        print(f"\n=== {loss_name} Results ===")
        print(f"Best Epoch: {best_epoch + 1}")
        print(f"Best Val Loss: {best_val_loss:.4f}")
        print(f"Loss Value: {metrics['loss_value']:.4f}")
        print(f"MSE: {metrics['mse']:.4f}")
        print(f"MAE: {metrics['mae']:.4f}")
        print(f"RMSE: {metrics['rmse']:.4f}")
        print(f"R²: {metrics['r2']:.4f}")
        print(f"MAPE: {metrics['mape']:.2f}%")
        print(f"Max Error: {metrics['max_error']:.4f}")
    
    # Create comprehensive results DataFrame
    print(f"\n{'='*60}")
    print("CREATING PREDICTIONS CSV")
    print(f"{'='*60}")
    
    # Start with validation data
    results_df = val_df.copy()
    
    # Add only predictions from each loss function
    for loss_name, predictions in all_predictions.items():
        results_df[f'predicted_{loss_name}'] = predictions
    
    # Save comprehensive results
    results_df.to_csv('all_loss_functions_predictions.csv', index=False)
    print(f"✓ All predictions saved to 'all_loss_functions_predictions.csv'")
    
    # Create metrics comparison
    metrics_comparison = pd.DataFrame({
        loss_name: {
            'Best_Epoch': training_info[loss_name]['best_epoch'] + 1,
            'Best_Val_Loss': training_info[loss_name]['best_val_loss'],
            'Loss_Value': all_metrics[loss_name]['loss_value'],
            'MSE': all_metrics[loss_name]['mse'],
            'MAE': all_metrics[loss_name]['mae'],
            'RMSE': all_metrics[loss_name]['rmse'],
            'R2': all_metrics[loss_name]['r2'],
            'MAPE': all_metrics[loss_name]['mape'],
            'Max_Error': all_metrics[loss_name]['max_error']
        }
        for loss_name in loss_functions.keys()
    }).T
    
    metrics_comparison.to_csv('loss_functions_comparison.csv')
    print(f"✓ Metrics comparison saved to 'loss_functions_comparison.csv'")
    
    # Print summary
    print(f"\n{'='*60}")
    print("LOSS FUNCTIONS COMPARISON SUMMARY")
    print(f"{'='*60}")
    print(metrics_comparison.round(4))
    
    # Find best performing loss function
    best_r2_idx = metrics_comparison['R2'].idxmax()
    best_mae_idx = metrics_comparison['MAE'].idxmin()
    best_mape_idx = metrics_comparison['MAPE'].idxmin()
    
    print(f"\n🏆 BEST PERFORMING LOSS FUNCTIONS:")
    print(f"  Best R² Score: {best_r2_idx} (R² = {metrics_comparison.loc[best_r2_idx, 'R2']:.4f})")
    print(f"  Best MAE: {best_mae_idx} (MAE = {metrics_comparison.loc[best_mae_idx, 'MAE']:.4f})")
    print(f"  Best MAPE: {best_mape_idx} (MAPE = {metrics_comparison.loc[best_mape_idx, 'MAPE']:.2f}%)")
    
    # Save the best model (based on R² score)
    print(f"\n💾 SAVING BEST MODEL...")
    best_loss_name = best_r2_idx
    
    # We need to retrain the best model to get its state
    print(f"Retraining best model ({best_loss_name}) to save...")
    
    # Create fresh model for the best loss function
    input_size = X_train.shape[1]
    best_model = SimpleNeuralNetwork(input_size=input_size, hidden_size=16, dropout_rate=0.1)
    best_loss_function = loss_functions[best_loss_name]
    
    # Train the best model
    train_losses, val_losses, best_epoch, best_val_loss = train_model_with_loss(
        best_model, X_train, y_train, X_val, y_val, best_loss_function, best_loss_name
    )
    
    # Save the best model
    torch.save({
        'model_state_dict': best_model.state_dict(),
        'loss_function_name': best_loss_name,
        'feature_columns': feature_columns,
        'metrics': all_metrics[best_loss_name],
        'training_info': training_info[best_loss_name],
        'best_epoch': best_epoch,
        'best_val_loss': best_val_loss,
        'model_architecture': {
            'input_size': input_size,
            'hidden_size': 16,
            'output_size': 1,
            'dropout_rate': 0.1
        }
    }, 'best_neural_network_model.pth')
    
    print(f"✓ Best model saved as 'best_neural_network_model.pth'")
    print(f"  Model: {best_loss_name}")
    print(f"  R² Score: {metrics_comparison.loc[best_r2_idx, 'R2']:.4f}")
    print(f"  MAE: {metrics_comparison.loc[best_r2_idx, 'MAE']:.4f}")
    print(f"  Best Epoch: {best_epoch + 1}")
    
    # Create visualization
    create_comparison_plot(all_predictions, results_df['anotacao'], metrics_comparison)
    
    print(f"\n📁 Files created:")
    print(f"  • all_loss_functions_predictions.csv - All predictions in one file")
    print(f"  • loss_functions_comparison.csv - Metrics comparison")
    print(f"  • loss_functions_comparison_plot.png - Visualization")
    print(f"  • best_neural_network_model.pth - Best performing model")
    
    return results_df, metrics_comparison

def create_comparison_plot(all_predictions, actual, metrics_comparison):
    """
    Create comparison visualization
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    loss_names = list(all_predictions.keys())
    
    for i, loss_name in enumerate(loss_names):
        predictions = all_predictions[loss_name]
        
        # Scatter plot
        axes[i].scatter(actual, predictions, alpha=0.7, s=50)
        axes[i].plot([actual.min(), actual.max()], [actual.min(), actual.max()], 'r--', lw=2)
        
        r2 = metrics_comparison.loc[loss_name, 'R2']
        mae = metrics_comparison.loc[loss_name, 'MAE']
        
        axes[i].set_xlabel('Actual Values')
        axes[i].set_ylabel('Predicted Values')
        axes[i].set_title(f'{loss_name}\nR² = {r2:.3f}, MAE = {mae:.2f}')
        axes[i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('loss_functions_comparison_plot.png', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    compare_loss_functions()

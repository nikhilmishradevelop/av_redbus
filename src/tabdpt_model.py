import pandas as pd
import polars as pl
import numpy as np
from datetime import datetime
import torch
import gc
import sys
import os
sys.path.append('../src')
sys.path.append('.')
from utils import Config, load_data, create_validation_splits, save_outputs

# TabDPT imports
from tabdpt import TabDPTRegressor
from sklearn.metrics import mean_squared_error
# StandardScaler no longer needed - normalization done at dataset level

def predict_fn(model, X_test, n_ensembles=2, context_size=256):
    """Predict function for TabDPT"""
    return model.predict(X_test, n_ensembles=4, context_size=1024*4)
    # return model.predict(X_test, n_ensembles=1, context_size=24)

def train_model(model_type, X_train, y_train, X_val, y_val, params, iterations):
    """Train a TabDPT model and return model, validation RMSE, and validation predictions"""
    
    print(f"Training {model_type} with {X_train.shape[0]} samples and {X_train.shape[1]} features...")
    
    # Convert to pandas DataFrame for TabDPT
    if isinstance(X_train, np.ndarray):
        feature_names = [f'feature_{i}' for i in range(X_train.shape[1])]
        X_train_df = pd.DataFrame(X_train, columns=feature_names)
        X_val_df = pd.DataFrame(X_val, columns=feature_names)
    else:
        X_train_df = X_train
        X_val_df = X_val
        feature_names = X_train_df.columns.tolist()
    
    # TabDPT expects numpy arrays
    X_train_np = X_train_df.values
    X_val_np = X_val_df.values
    y_train_np = y_train.copy()
    y_val_np = y_val.copy()
    
    # Handle potential infinity/NaN values
    X_train_np = np.nan_to_num(X_train_np, nan=0.0, posinf=1e6, neginf=-1e6)
    X_val_np = np.nan_to_num(X_val_np, nan=0.0, posinf=1e6, neginf=-1e6)
    y_train_np = np.nan_to_num(y_train_np, nan=0.0, posinf=1e6, neginf=-1e6)
    y_val_np = np.nan_to_num(y_val_np, nan=0.0, posinf=1e6, neginf=-1e6)
    
    # Features are already normalized at dataset level
    
    # Create TabDPT model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = TabDPTRegressor(
        device=device,
        inf_batch_size=params.get('batch_size', 128)
    )
    
    # Fit the model
    print(f"Training TabDPT on {device}...")
    model.fit(X_train_np, y_train_np)
    
    # Get validation predictions
    
    # Print all the attributes of the model
    # print(model.__dict__)
    
    val_pred = predict_fn(model, X_val_np, n_ensembles=2, context_size=256)
    val_rmse = np.sqrt(mean_squared_error(y_val_np, val_pred))
    
    
    # No scaler needed - normalization done at dataset level
    
    return model, val_rmse, val_pred

def train_final_model(model_type, X_all, y_all, params, iterations):
    """Train final TabDPT model on all data for submission"""
    
    print(f"Training final {model_type} with {X_all.shape[0]} samples and {X_all.shape[1]} features...")
    
    # Convert to numpy if needed
    if isinstance(X_all, pd.DataFrame):
        X_all_np = X_all.values
    else:
        X_all_np = X_all
    
    y_all_np = y_all.copy()
    
    # Handle potential infinity/NaN values
    X_all_np = np.nan_to_num(X_all_np, nan=0.0, posinf=1e6, neginf=-1e6)
    y_all_np = np.nan_to_num(y_all_np, nan=0.0, posinf=1e6, neginf=-1e6)
    
    # Features are already normalized at dataset level
    
    # Create TabDPT model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = TabDPTRegressor(
        device=device,
        inf_batch_size=params.get('batch_size', 128)
    )
    
    # Fit the model (no validation set for final training)
    print(f"Training final TabDPT on {device}...")
    model.fit(X_all_np, y_all_np)
    
    return model

def predict_model(model_type, model, X_test):
    """Get predictions from TabDPT model"""
    
    # Convert to numpy if needed
    if isinstance(X_test, pd.DataFrame):
        X_test_np = X_test.values
    else:
        X_test_np = X_test
    
    # Handle potential infinity/NaN values
    X_test_np = np.nan_to_num(X_test_np, nan=0.0, posinf=1e6, neginf=-1e6)
    
    # Features are already normalized at dataset level
    # Get predictions
    predictions = predict_fn(model, X_test_np, n_ensembles=2, context_size=256)
    
    return predictions

def run_experiment_with_features(X_train_pl, X_val_pl, X_test_pl, y_train_np, y_val_np, y_test_np, 
                                feature_cols, model_config, experiment_name, run_validation=False):
    """Run training experiment with given features and TabDPT model - validation optional"""
    
    print(f"\n{'='*60}")
    print(f"EXPERIMENT: {experiment_name}")
    print(f"Model: {model_config['type']}")
    print(f"Using {len(feature_cols)} features")
    print(f"Validation: {'ON' if run_validation else 'OFF'}")
    print(f"{'='*60}")
    
    # Convert to numpy for TabDPT
    X_train = X_train_pl.select(feature_cols).to_numpy()
    X_test = X_test_pl.select(feature_cols).to_numpy()
    
    # Handle potential infinity/NaN values
    X_train = np.nan_to_num(X_train, nan=0.0, posinf=1e6, neginf=-1e6)
    X_test = np.nan_to_num(X_test, nan=0.0, posinf=1e6, neginf=-1e6)
    
    val_rmse, val_predictions = None, None
    
    if run_validation:
        X_val = X_val_pl.select(feature_cols).to_numpy()
        X_val = np.nan_to_num(X_val, nan=0.0, posinf=1e6, neginf=-1e6)
        
        # Train model and get validation predictions
        model, val_rmse, val_predictions = train_model(
            model_config['type'], X_train, y_train_np, X_val, y_val_np, 
            model_config['params'], model_config['iterations']
        )
        
        # Train on train+val for better test predictions
        print("Training on train+val for holdout test predictions...")
        X_train_val = np.vstack([X_train, X_val])
        y_train_val = np.concatenate([y_train_np, y_val_np])
        
        del X_val, model
    else:
        # Skip validation, train directly on train+val
        print("Skipping validation, training on train+val...")
        X_val = X_val_pl.select(feature_cols).to_numpy()
        X_val = np.nan_to_num(X_val, nan=0.0, posinf=1e6, neginf=-1e6)
        X_train_val = np.vstack([X_train, X_val])
        y_train_val = np.concatenate([y_train_np, y_val_np])
        
        del X_val
    
    # Train model on combined train+val data
    train_val_model = train_final_model(model_config['type'], X_train_val, y_train_val, 
                                       model_config['params'], model_config['iterations'])
    
    # Get test predictions from train+val model
    test_predictions = predict_model(model_config['type'], train_val_model, X_test)
    test_rmse = np.sqrt(mean_squared_error(y_test_np, test_predictions))
    
    # Clean up train+val model
    del X_train_val, y_train_val, train_val_model
    
    if run_validation:
        print(f"Validation RMSE: {val_rmse:.2f}")
    print(f"Test RMSE: {test_rmse:.2f}")
    
    # Clean up
    del X_train, X_test
    gc.collect()
    
    return val_rmse, test_rmse, val_predictions, test_predictions

def get_model_configs(quick_mode=False):
    """Define TabDPT model configurations - easily configurable"""
    
    # TabDPT is a pre-trained model, so we only need inference parameters
    batch_size = 256 if quick_mode else 256
    
    tabdpt_config = {
        'type': 'tabdpt',
        'params': {
            'batch_size': batch_size,  # Inference batch size
        },
        'iterations': 1  # Not used for TabDPT but kept for consistency
    }
    
    return {
        'tabdpt': tabdpt_config,
    }

def get_feature_configs():
    """Define feature configurations for different experiments"""
    return {
        'top_1k': {'n_features': 1000, 'name': 'Top 1K Features'},
        'top_2k': {'n_features': 2000, 'name': 'Top 2K Features'},
        'top_3k': {'n_features': 3000, 'name': 'Top 3K Features'},
        'top_6k': {'n_features': 6000, 'name': 'Top 6K Features'},
    }

def ensemble_predictions(predictions_list, weights=None):
    """Ensemble multiple predictions with optional weights"""
    if weights is None:
        weights = [1.0 / len(predictions_list)] * len(predictions_list)
    
    ensemble_pred = np.zeros_like(predictions_list[0])
    for pred, weight in zip(predictions_list, weights):
        ensemble_pred += weight * pred
    
    return ensemble_pred

def main():
    config = Config(version='tabdpt_model')
    
    # CONFIGURATION OPTIONS
    RUN_VALIDATION = False  # Set to True to run validation (slower)
    QUICK_MODE = False      # Use smaller dataset for quick testing
    QUICK_SAMPLES = 10000   # Samples for quick mode
    TARGET_RMSE = 600       # Target RMSE
    RUN_ENSEMBLE = True     # Run ensemble of multiple feature sets
    
    # Check GPU availability
    if torch.cuda.is_available():
        print(f"CUDA available! Using GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print("WARNING: CUDA not available, using CPU (will be slow)")
    
    if QUICK_MODE:
        print(f"QUICK EXPERIMENT MODE: Using {QUICK_SAMPLES} samples to target RMSE < {TARGET_RMSE}")
    
    print("Loading optimized dataset with top features for TabDPT...")
    
    # Load the optimized dataset
    optimized_file = '../outputs/optimized_dataset_top6k_features.ipc'
    if not os.path.exists(optimized_file):
        print(f"ERROR: Optimized dataset not found at {optimized_file}")
        return
    
    df = pl.read_ipc(optimized_file)
    print(f"Loaded optimized dataset: {df.shape}")
    
    # Normalize the entire dataset at once to prevent data leakage
    print("Normalizing entire dataset...")
    exclude_cols = ['doj', 'route', 'final_seatcount', 'srcid', 'destid', 'year']
    numeric_cols = [col for col in df.columns 
                   if col not in exclude_cols and 
                   str(df[col].dtype) not in ['Date', 'Datetime', 'Utf8', 'Boolean']]
    
    # Apply normalization using polars with_columns for all numeric columns at once
    df = df.with_columns([
        ((pl.col(col) - pl.col(col).mean()) / (pl.col(col).std() + 1e-8)).alias(col)
        for col in numeric_cols
    ])
    
    print(f"Normalized {len(numeric_cols)} numeric columns")
    
    # Load feature importance
    feature_importance_file = '../outputs/feature_importance.csv'
    if not os.path.exists(feature_importance_file):
        print(f"ERROR: Feature importance file not found at {feature_importance_file}")
        return
        
    importance_df = pd.read_csv(feature_importance_file)
    print("Loaded feature importance")
    
    # Separate train and test data
    train_df = df.filter(pl.col('final_seatcount').is_not_null())
    test_df = df.filter(pl.col('final_seatcount').is_null())
    
    print(f"Train data: {train_df.shape}")  
    print(f"Test data: {test_df.shape}")
    
    del df
    gc.collect()
    
    train_mask, val_mask, test_mask, _ = create_validation_splits(train_df)
    
    exclude_cols = ['doj', 'route', 'final_seatcount', 'srcid', 'destid', 'year']
    feature_cols = [col for col in train_df.columns 
                   if col not in exclude_cols and 
                   str(train_df[col].dtype) not in ['Date', 'Datetime']]
    
    # Get feature configurations
    feature_configs = get_feature_configs()
    available_features = [f for f in importance_df['feature'].tolist() if f in feature_cols]
    print(f"Available features from importance: {len(available_features)}")
    
    # Prepare feature sets for different configurations
    feature_sets = {}
    for config_name, feature_config in feature_configs.items():
        n_features = min(feature_config['n_features'], len(available_features))
        features = available_features[:n_features]
        feature_sets[config_name] = {
            'features': features,
            'name': feature_config['name'],
            'count': len(features)
        }
        print(f"{feature_config['name']}: {len(features)} features")
    
    # Extract target
    y_np = train_df.select('final_seatcount').to_numpy().flatten()
    y_train, y_val, y_test = y_np[train_mask], y_np[val_mask], y_np[test_mask]
    
    # Keep train_df in polars format for memory efficiency
    X_train_pl = train_df.filter(pl.Series(train_mask))
    X_val_pl = train_df.filter(pl.Series(val_mask))
    X_test_pl = train_df.filter(pl.Series(test_mask))
    
    # QUICK MODE: Use smaller datasets for faster experimentation
    if QUICK_MODE:
        print(f"Sampling {QUICK_SAMPLES} training samples for quick experiments...")
        
        # Sample from training data
        if len(X_train_pl) > QUICK_SAMPLES:
            train_indices = np.random.choice(len(X_train_pl), size=QUICK_SAMPLES, replace=False)
            X_train_pl = X_train_pl[train_indices]
            y_train = y_train[train_indices]
        
        print(f"Quick mode: Train={len(X_train_pl)}, Val={len(X_val_pl)}, Test={len(X_test_pl)} samples")
    
    del y_np, train_mask, val_mask, test_mask
    gc.collect()
    
    # Get model configurations
    model_configs = get_model_configs(quick_mode=QUICK_MODE)
    
    print(f"\n{'='*60}")
    print("RUNNING TABDPT EXPERIMENTS")
    print(f"Feature sets: {list(feature_sets.keys())}")
    print(f"Validation: {'ON' if RUN_VALIDATION else 'OFF'}")
    print(f"Ensemble: {'ON' if RUN_ENSEMBLE else 'OFF'}")
    print(f"{'='*60}")
    
    experiment_results = []
    model_config = model_configs['tabdpt']
    
    # Run experiments for each feature set
    for config_name, feature_set in feature_sets.items():
        experiment_name = f"TabDPT {feature_set['name']}"
        
        val_rmse, test_rmse, val_preds, test_preds = run_experiment_with_features(
            X_train_pl, X_val_pl, X_test_pl, y_train, y_val, y_test,
            feature_set['features'], model_config, experiment_name, RUN_VALIDATION
        )
        
        print(f"RMSE: {test_rmse:.2f} (target: <{TARGET_RMSE})")
        if test_rmse > TARGET_RMSE:
            print(f"WARNING: RMSE {test_rmse:.2f} > target {TARGET_RMSE}")
        else:
            print(f"SUCCESS: RMSE {test_rmse:.2f} < target {TARGET_RMSE}")
        
        experiment_results.append({
            'name': experiment_name,
            'model_type': 'tabdpt',
            'feature_config': config_name,
            'features': feature_set['features'],
            'feature_count': feature_set['count'],
            'val_rmse': val_rmse,
            'test_rmse': test_rmse,
            'val_predictions': val_preds,
            'test_predictions': test_preds,
            'config': model_config
        })
    
    # Ensemble predictions if enabled
    if RUN_ENSEMBLE and len(experiment_results) > 1:
        print(f"\n{'='*60}")
        print("CREATING ENSEMBLE")
        print(f"{'='*60}")
        
        test_predictions_list = [result['test_predictions'] for result in experiment_results]
        ensemble_test_preds = ensemble_predictions(test_predictions_list)
        ensemble_test_rmse = np.sqrt(mean_squared_error(y_test, ensemble_test_preds))
        
        val_predictions_list = None
        ensemble_val_preds = None
        ensemble_val_rmse = None
        
        if RUN_VALIDATION:
            val_predictions_list = [result['val_predictions'] for result in experiment_results if result['val_predictions'] is not None]
            if val_predictions_list:
                ensemble_val_preds = ensemble_predictions(val_predictions_list)
                ensemble_val_rmse = np.sqrt(mean_squared_error(y_val, ensemble_val_preds))
        
        # Add ensemble result
        experiment_results.append({
            'name': 'Ensemble (All Features)',
            'model_type': 'ensemble',
            'feature_config': 'ensemble',
            'features': None,
            'feature_count': None,
            'val_rmse': ensemble_val_rmse,
            'test_rmse': ensemble_test_rmse,
            'val_predictions': ensemble_val_preds,
            'test_predictions': ensemble_test_preds,
            'config': None
        })
        
        if RUN_VALIDATION:
            print(f"Ensemble Validation RMSE: {ensemble_val_rmse:.2f}")
        print(f"Ensemble Test RMSE: {ensemble_test_rmse:.2f}")
    
    # Display results
    print(f"\n{'='*60}")
    print("TABDPT EXPERIMENT RESULTS")
    print(f"{'='*60}")
    for result in experiment_results:
        val_str = f"Val RMSE: {result['val_rmse']:.2f}, " if result['val_rmse'] is not None else ""
        feat_str = f"({result['feature_count']} features)" if result['feature_count'] is not None else ""
        print(f"{result['name']:25s} - {val_str}Test RMSE: {result['test_rmse']:.2f} {feat_str}")
    
    # Train TabDPT for final submission (skip in quick mode)
    if not QUICK_MODE:
        print(f"\n{'='*60}")
        print("TRAINING TABDPT FOR FINAL SUBMISSION")
        print(f"{'='*60}")
        
        _, test_df_original, _ = load_data()
        test_df_final = test_df.with_columns([
                pl.col('doj').cast(pl.Date)
            ]).join(
            test_df_original.select(['doj', 'srcid', 'destid', 'route_key']).with_columns([
                pl.col('doj').cast(pl.Date)
            ]),
            on=['doj', 'srcid', 'destid'],
            how='left'
        )
        
        del test_df_original
        gc.collect()
        
        # Train ensemble of all feature configurations for submission
        print(f"\nTraining ensemble of all feature configurations for submission...")
        submission_predictions_list = []
        
        # Train models for each feature configuration
        for result in experiment_results:
            if result['model_type'] == 'tabdpt':  # Skip ensemble result
                print(f"Training {result['name']} for final submission...")
                features = result['features']
                y_all_np = train_df.select('final_seatcount').to_numpy().flatten()
                
                X_all = train_df.select(features).to_numpy()
                X_test_final = test_df_final.select(features).to_numpy()
                
                # Handle infinity values
                X_all = np.nan_to_num(X_all, nan=0.0, posinf=1e6, neginf=-1e6)
                X_test_final = np.nan_to_num(X_test_final, nan=0.0, posinf=1e6, neginf=-1e6)
                
                model = train_final_model('tabdpt', X_all, y_all_np, model_config['params'], model_config['iterations'])
                pred = predict_model('tabdpt', model, X_test_final)
                submission_predictions_list.append(pred)
                
                del X_all, X_test_final, y_all_np, model
                gc.collect()
        
        # Ensemble all predictions
        if len(submission_predictions_list) > 1:
            test_predictions = ensemble_predictions(submission_predictions_list)
            features_used = 'ensemble_all_feature_configs'
            print(f"Ensembled {len(submission_predictions_list)} models for final submission")
        else:
            test_predictions = submission_predictions_list[0]
            features_used = 'single_model'
            print("Using single model for final submission")
        
        # Save outputs - ensure config is the Config object, not a dict
        save_outputs(None, test_df_final, test_predictions, features_used, config)
    
    print(f"\n{'='*60}")
    if QUICK_MODE:
        print("QUICK TABDPT EXPERIMENT COMPLETED")
    else:
        print("TABDPT TRAINING COMPLETED")
    print(f"{'='*60}")
    if experiment_results:
        result = experiment_results[0]
        if QUICK_MODE:
            print(f"{result['name']} Test RMSE: {result['test_rmse']:.2f} (target: <{TARGET_RMSE})")
            best_result = min(experiment_results, key=lambda x: x['test_rmse'])
            if best_result['test_rmse'] < TARGET_RMSE:
                print("✓ Ready for full training! Set QUICK_MODE = False to train on full dataset.")
            else:
                print("⚠ Consider tuning hyperparameters before full training.")
            print(f"Best model: {best_result['name']} with Test RMSE: {best_result['test_rmse']:.2f}")
        else:
            best_result = min(experiment_results, key=lambda x: x['test_rmse'])
            if best_result['val_rmse'] is not None:
                print(f"Best {best_result['name']} Val RMSE: {best_result['val_rmse']:.2f}")
            print(f"Best {best_result['name']} Test RMSE: {best_result['test_rmse']:.2f}")
            print(f"Submission saved to: {config.get_submission_path()}")
            if RUN_ENSEMBLE:
                ensemble_result = [r for r in experiment_results if r['model_type'] == 'ensemble'][0]
                print(f"Ensemble used for final submission with Test RMSE: {ensemble_result['test_rmse']:.2f}")

if __name__ == "__main__":
    main()
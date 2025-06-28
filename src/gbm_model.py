import pandas as pd
import polars as pl
import numpy as np
from datetime import datetime
import lightgbm as lgb
import catboost as cb
import xgboost as xgb
from sklearn.metrics import mean_squared_error
import gc
import os
import pickle
from utils import Config, load_data, create_validation_splits, save_outputs

def load_optimized_hyperparameters(model_type):
    """Load the best hyperparameters from hyperparameter optimization"""
    print(f"Using optimized {model_type} hyperparameters")
    return get_optimized_params(model_type)

def get_optimized_params(model_type):
    """Optimized hyperparameters from hyperparameter optimization"""
    if model_type == 'lightgbm':
        return {
            'learning_rate': 0.02714781731750764,
            'num_leaves': 46,
            'max_depth': 8,
            'min_data_in_leaf': 5,
            'feature_fraction': 0.7853350391879671,
            'bagging_fraction': 0.5783262863967494,
            'bagging_freq': 3,
            'lambda_l1': 55.75202548057786,
            'lambda_l2': 22.820036780583614,
            'min_gain_to_split': 6.061076800213592,
            'max_bin': 107,
            'iterations': 1144
        }
    elif model_type == 'catboost':
        return {
            'learning_rate': 0.22676764522798332,
            'depth': 6,
            'l2_leaf_reg': 5.036532286983366,
            'random_strength': 1.467355689442534,
            'bootstrap_type': 'Bernoulli',
            'subsample': 0.8780976501589233,
            'rsm': 0.43230812789633855,
            'border_count': 243,
            'iterations': 625
        }
    elif model_type == 'xgboost':
        return {
            'eta': 0.05502641736603284,
            'max_depth': 5,
            'min_child_weight': 8.993152841507277,
            'subsample': 0.6182855588845002,
            'colsample_bytree': 0.5658605114261004,
            'colsample_bylevel': 0.7079876516516623,
            'reg_alpha': 44.63220084500361,
            'reg_lambda': 97.52947027803984,
            'gamma': 1.389776656070271,
            'iterations': 1803
        }
    else:
        # Fallback to default params
        return get_default_params(model_type)

def get_default_params(model_type):
    """Default parameters if optimization results are not available"""
    if model_type == 'lightgbm':
        return {
            'learning_rate': 0.03,
            'num_leaves': 105,
            'max_depth': -1,
            'min_data_in_leaf': 30,
            'feature_fraction': 0.45,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'lambda_l1': 10,
            'lambda_l2': 10,
            'min_gain_to_split': 0,
            'max_bin': 25,
            'iterations': 1000
        }
    elif model_type == 'xgboost':
        return {
            'eta': 0.05,
            'max_depth': 5,
            'min_child_weight': 1,
            'subsample': 0.8,
            'colsample_bytree': 0.3,
            'colsample_bylevel': 1.0,
            'reg_alpha': 10,
            'reg_lambda': 10,
            'gamma': 0,
            'iterations': 1000
        }
    elif model_type == 'catboost':
        return {
            'learning_rate': 0.05,
            'depth': 5,
            'rsm': 0.3,
            'l2_leaf_reg': 3,
            'random_strength': 1,
            'bootstrap_type': 'Bayesian',
            'bagging_temperature': 1,
            'iterations': 1000
        }

def train_model(model_type, X_train, y_train, X_val, y_val, params):
    """Train a model and return model, validation RMSE, and validation predictions"""
    
    # Extract iterations from params
    iterations = params.pop('iterations', 1000)
    
    if model_type == 'lightgbm':
        # Prepare LightGBM specific params
        lgb_params = {
            'objective': 'regression',
            'metric': 'rmse',
            'verbosity': -1,
            'random_state': 42,
            **params
        }
        
        model = lgb.train(
            lgb_params,
            lgb.Dataset(X_train, label=y_train),
            valid_sets=[lgb.Dataset(X_val, label=y_val)],
            num_boost_round=iterations,
            callbacks=[lgb.log_evaluation(0)]
        )
        val_pred = model.predict(X_val)
        
    elif model_type == 'catboost':
        # Prepare CatBoost specific params
        cb_params = {
            'loss_function': 'RMSE',
            'random_seed': 42,
            'verbose': False,
            'allow_writing_files': False,
            'task_type': 'CPU',
            **params
        }
        
        model = cb.CatBoostRegressor(**cb_params, iterations=iterations)
        model.fit(X_train, y_train, eval_set=(X_val, y_val), verbose=50)
        val_pred = model.predict(X_val)
        
    elif model_type == 'xgboost':
        # Handle potential infinity/NaN values for XGBoost
        X_train_clean = np.nan_to_num(X_train, nan=0.0, posinf=1e6, neginf=-1e6)
        X_val_clean = np.nan_to_num(X_val, nan=0.0, posinf=1e6, neginf=-1e6)
        
        # Prepare XGBoost specific params
        xgb_params = {
            'objective': 'reg:squarederror',
            'eval_metric': 'rmse',
            'random_state': 42,
            'verbosity': 0,
            **params
        }
        
        dtrain = xgb.DMatrix(X_train_clean, label=y_train)
        dval = xgb.DMatrix(X_val_clean, label=y_val)
        model = xgb.train(
            xgb_params, dtrain,
            num_boost_round=iterations,
            evals=[(dval, 'eval')],
            verbose_eval=50
        )
        val_pred = model.predict(dval)
        
    val_rmse = np.sqrt(mean_squared_error(y_val, val_pred))
    return model, val_rmse, val_pred

def train_final_model(model_type, X_all, y_all, params):
    """Train final model on all data for submission"""
    
    # Extract iterations from params
    iterations = params.pop('iterations', 1000)
    
    if model_type == 'lightgbm':
        lgb_params = {
            'objective': 'regression',
            'metric': 'rmse',
            'verbosity': -1,
            'random_state': 42,
            **params
        }
        
        model = lgb.train(
            lgb_params,
            lgb.Dataset(X_all, label=y_all),
            num_boost_round=iterations,
            callbacks=[lgb.log_evaluation(0)]
        )
        
    elif model_type == 'catboost':
        cb_params = {
            'loss_function': 'RMSE',
            'random_seed': 42,
            'verbose': False,
            'allow_writing_files': False,
            'task_type': 'CPU',
            **params
        }
        
        model = cb.CatBoostRegressor(**cb_params, iterations=iterations)
        model.fit(X_all, y_all, verbose=50)
        
    elif model_type == 'xgboost':
        # Handle potential infinity/NaN values for XGBoost
        X_all_clean = np.nan_to_num(X_all, nan=0.0, posinf=1e6, neginf=-1e6)
        
        xgb_params = {
            'objective': 'reg:squarederror',
            'eval_metric': 'rmse',
            'random_state': 42,
            'verbosity': 0,
            **params
        }
        
        dtrain = xgb.DMatrix(X_all_clean, label=y_all)
        model = xgb.train(xgb_params, dtrain, num_boost_round=iterations, verbose_eval=50)
        
    return model

def predict_model(model_type, model, X_test):
    """Get predictions from model"""
    if model_type == 'xgboost':
        # Handle potential infinity/NaN values for XGBoost
        X_test_clean = np.nan_to_num(X_test, nan=0.0, posinf=1e6, neginf=-1e6)
        dtest = xgb.DMatrix(X_test_clean)
        return model.predict(dtest)
    else:
        return model.predict(X_test)

def run_experiment_with_optimized_params(X_train_pl, X_val_pl, X_test_pl, y_train_np, y_val_np, y_test_np, 
                                       feature_cols, model_type, experiment_name, feature_fraction_multiplier=1.0):
    """Run training experiment with optimized hyperparameters"""
    
    print(f"\n{'='*60}")
    print(f"EXPERIMENT: {experiment_name}")
    print(f"Model: {model_type}")
    print(f"Using {len(feature_cols)} features")
    print(f"Feature fraction multiplier: {feature_fraction_multiplier}")
    print(f"{'='*60}")
    
    # Load optimized hyperparameters
    optimized_params = load_optimized_hyperparameters(model_type)
    
    # Adjust feature fraction parameters if multiplier is not 1.0
    if feature_fraction_multiplier != 1.0:
        optimized_params = adjust_feature_fraction_params(optimized_params, feature_fraction_multiplier)
        print(f"Adjusted feature fraction parameters by {feature_fraction_multiplier}x")
    
    # Convert to numpy/pandas as needed
    if model_type == 'catboost':
        X_train = X_train_pl.select(feature_cols).to_pandas()
        X_val = X_val_pl.select(feature_cols).to_pandas()
        X_test = X_test_pl.select(feature_cols).to_pandas()
    else:
        X_train = X_train_pl.select(feature_cols).to_numpy()
        X_val = X_val_pl.select(feature_cols).to_numpy()
        X_test = X_test_pl.select(feature_cols).to_numpy()
    
    # Train model and get validation predictions
    model, val_rmse, val_predictions = train_model(
        model_type, X_train, y_train_np, X_val, y_val_np, optimized_params.copy()
    )
    
    # Combine train and val for final training on test set
    if model_type == 'catboost':
        X_train_val = pd.concat([X_train, X_val], axis=0)
    else:
        X_train_val = np.vstack([X_train, X_val])
    y_train_val = np.concatenate([y_train_np, y_val_np])
    
    final_model = train_final_model(
        model_type, X_train_val, y_train_val, optimized_params.copy()
    )
    
    # Get test predictions
    test_predictions = predict_model(model_type, final_model, X_test)
    test_rmse = np.sqrt(mean_squared_error(y_test_np, test_predictions))
    
    print(f"Validation RMSE: {val_rmse:.2f}")
    print(f"Test RMSE: {test_rmse:.2f}")
    
    # Clean up
    del X_train, X_val, X_test, X_train_val, y_train_val, model, final_model
    gc.collect()
    
    return val_rmse, test_rmse, val_predictions, test_predictions, optimized_params

# =============================================================================
# CONFIGURATION SECTION - EASILY CONFIGURABLE
# =============================================================================

# Feature configurations - add/remove as needed
FEATURE_CONFIGS = [
    {
        'name': '1K_features',
        'top_n_features': 1000,
        'feature_fraction_multiplier': 1.0  # Use original feature fraction
    },
    # Top 2K features
    {
        'name': '2K_features',
        'top_n_features': 2000,
        'feature_fraction_multiplier': 0.85  # Use original feature fraction
    },
    # Top 3K features
    {
        'name': '3K_features', 
        'top_n_features': 3000,
        'feature_fraction_multiplier': 0.7  # 60% of original feature fraction
    },
    # Add more configurations here:
    {
        'name': '6K_features',
        'top_n_features': 6000, 
        'feature_fraction_multiplier': 0.4
    },
]

# Model configurations - add/remove as needed
MODELS_TO_RUN = ['lightgbm', 'xgboost', 'catboost']

# =============================================================================

def adjust_feature_fraction_params(params, multiplier):
    """Adjust feature fraction parameters by multiplier"""
    adjusted_params = params.copy()
    
    # LightGBM uses 'feature_fraction'
    if 'feature_fraction' in adjusted_params:
        adjusted_params['feature_fraction'] = min(1.0, adjusted_params['feature_fraction'] * multiplier)
    
    # XGBoost uses 'colsample_bytree' and 'colsample_bylevel'  
    if 'colsample_bytree' in adjusted_params:
        adjusted_params['colsample_bytree'] = min(1.0, adjusted_params['colsample_bytree'] * multiplier)
    if 'colsample_bylevel' in adjusted_params:
        adjusted_params['colsample_bylevel'] = min(1.0, adjusted_params['colsample_bylevel'] * multiplier)
    
    # CatBoost uses 'rsm' (random subspace method)
    if 'rsm' in adjusted_params:
        adjusted_params['rsm'] = min(1.0, adjusted_params['rsm'] * multiplier)
    
    return adjusted_params

def main():
    config = Config(version='v17_filtered')
    
    print("Loading optimized dataset with top 6000 features...")
    
    # Load the optimized dataset
    optimized_file = '../outputs/optimized_dataset_top6k_features.ipc'
    if not os.path.exists(optimized_file):
        print(f"ERROR: Optimized dataset not found at {optimized_file}")
        return
    
    df = pl.read_ipc(optimized_file)
    print(f"Loaded optimized dataset: {df.shape}")
    
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
    
    # Problematic features from validation analysis
    problematic_features = [
        'days_since_route_start',
        'quarter_week_interaction',
        'trans_cumsum_searchcount_std',
        'trans_seatcount_diff_sum',
        'route_dbd_searchcount_acceleration_q95_dbd_15_18',
        'dest_weekday_search_seat_diff_ratio_mean',
        'searchcount_mean_by_route_doi_median_60d_min',
        'dest_month_seatcount_lag3_q95',
        'src_seatcount_median_by_route_doi_q95',
        'search_intensity_per_day_mean_pct7',
        'src_dest_region_is_holiday_searchcount_std_by_route_doi_mean',
        'route_dbd_seatcount_lag1_sum_dbd_26_31',
        'src_dest_region_year_week_searchcount_acceleration_std',
        'src_dest_region_year_week_seatcount_acceleration_min',
        'searchcount_acceleration_std_lag8',
        'route_dbd_seatcount_rank_by_route_max_dbd_26_31',
        'route_dbd_searchcount_growth_rate_q95_dbd_26_31',
        'route_dbd_seatcount_diff_cummean_sum_dbd_21_26',
        'booking_search_accel_ratio_max_90d_std',
        'searchcount_lag1_std_5d_std'
    ]
    
    print(f"Excluding {len(problematic_features)} problematic features from validation analysis")
    
    # Exclude temporal features (from v16) + problematic features from validation analysis
    exclude_cols = ['doj', 'route', 'final_seatcount', 'srcid', 'destid', 'year', 'day_of_year', 'week'] + problematic_features
    
    feature_cols = [col for col in train_df.columns 
                   if col not in exclude_cols and 
                   str(train_df[col].dtype) not in ['Date', 'Datetime']]
    
    print("Excluded temporal features: day_of_year, week, year")
    print(f"Excluded problematic features from validation analysis: {len(problematic_features)}")
    print(f"Total features before exclusion: {len(train_df.columns)}")
    print(f"Features after exclusion: {len(feature_cols)}")
    
    # Extract target
    y_np = train_df.select('final_seatcount').to_numpy().flatten()
    y_train, y_val, y_test = y_np[train_mask], y_np[val_mask], y_np[test_mask]
    
    # Keep train_df in polars format for memory efficiency
    X_train_pl = train_df.filter(pl.Series(train_mask))
    X_val_pl = train_df.filter(pl.Series(val_mask))
    X_test_pl = train_df.filter(pl.Series(test_mask))
    
    del y_np, train_mask, val_mask, test_mask
    gc.collect()
    
    print(f"\n{'='*80}")
    print("RUNNING FILTERED EXPERIMENTS (V17)")
    print(f"Feature Configs: {[cfg['name'] for cfg in FEATURE_CONFIGS]}")
    print(f"Models: {MODELS_TO_RUN}")
    print(f"Total Experiments: {len(FEATURE_CONFIGS)} × {len(MODELS_TO_RUN)} = {len(FEATURE_CONFIGS) * len(MODELS_TO_RUN)}")
    print(f"{'='*80}")
    
    experiment_results = []
    
    # Run experiments for each feature configuration
    for feature_config in FEATURE_CONFIGS:
        feature_name = feature_config['name']
        top_n = feature_config['top_n_features']
        feature_fraction_mult = feature_config['feature_fraction_multiplier']
        
        # Get top N features for this configuration, excluding temporal and problematic features
        top_features = importance_df.head(top_n)['feature'].tolist()
        top_features = [f for f in top_features if f in feature_cols]
        
        print(f"\n{'='*60}")
        print(f"FEATURE SET: {feature_name} (Top {top_n} features)")
        print(f"Available features after all exclusions: {len(top_features)}")
        print(f"Feature fraction multiplier: {feature_fraction_mult}")
        print(f"{'='*60}")
        
        # Run each model with this feature set
        for model_name in MODELS_TO_RUN:
            experiment_name = f"{model_name.upper()}_{feature_name}"
            
            val_rmse, test_rmse, val_preds, test_preds, used_params = run_experiment_with_optimized_params(
                X_train_pl, X_val_pl, X_test_pl, y_train, y_val, y_test,
                top_features, model_name, experiment_name, feature_fraction_mult
            )
            
            experiment_results.append({
                'name': experiment_name,
                'model_type': model_name,
                'feature_config': feature_name,
                'features': top_features,
                'feature_count': len(top_features),
                'feature_fraction_multiplier': feature_fraction_mult,
                'val_rmse': val_rmse,
                'test_rmse': test_rmse,
                'val_predictions': val_preds,
                'test_predictions': test_preds,
                'optimized_params': used_params
            })
    
    # Calculate ensemble predictions
    print(f"\n{'='*60}")
    print("CALCULATING FILTERED ENSEMBLE PREDICTIONS")
    print(f"{'='*60}")
    
    ensemble_val_preds = np.mean([result['val_predictions'] for result in experiment_results], axis=0)
    ensemble_val_rmse = np.sqrt(mean_squared_error(y_val, ensemble_val_preds))
    
    ensemble_test_preds = np.mean([result['test_predictions'] for result in experiment_results], axis=0)
    ensemble_test_rmse = np.sqrt(mean_squared_error(y_test, ensemble_test_preds))
    
    # Select best individual configuration
    best_config = min(experiment_results, key=lambda x: x['val_rmse'])
    
    print(f"\n{'='*80}")
    print("FILTERED EXPERIMENT RESULTS")
    print(f"{'='*80}")
    
    # Group results by feature configuration for better readability
    for feature_config in FEATURE_CONFIGS:
        feature_name = feature_config['name']
        config_results = [r for r in experiment_results if r['feature_config'] == feature_name]
        
        print(f"\n{feature_name} Results:")
        for result in config_results:
            print(f"  {result['name']:25s} - Val: {result['val_rmse']:.2f}, Test: {result['test_rmse']:.2f}")
    
    print(f"\n{'='*80}")
    print("FILTERED ENSEMBLE RESULTS")
    print(f"{'='*80}")
    print(f"Ensemble Val RMSE:  {ensemble_val_rmse:.2f}")
    print(f"Ensemble Test RMSE: {ensemble_test_rmse:.2f}")
    print(f"Total models in ensemble: {len(experiment_results)}")
    
    print(f"\n{'='*80}")
    print("COMPARISON: BEST INDIVIDUAL vs ENSEMBLE")
    print(f"{'='*80}")
    print(f"Best Individual ({best_config['name']:25s}) - Val: {best_config['val_rmse']:.2f}, Test: {best_config['test_rmse']:.2f}")
    print(f"Ensemble ({len(experiment_results)} models avg)       - Val: {ensemble_val_rmse:.2f}, Test: {ensemble_test_rmse:.2f}")
    
    # Train ensemble for final submission
    print(f"\n{'='*60}")
    print("TRAINING FILTERED ENSEMBLE FOR FINAL SUBMISSION")
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
    
    # Train all models for ensemble submission
    ensemble_predictions = []
    
    model_count = 0
    total_models = len(experiment_results)
    
    # Train each model from the experiment results
    for result in experiment_results:
        model_count += 1
        model_name = result['model_type']
        feature_list = result['features']
        feature_mult = result['feature_fraction_multiplier']
        
        print(f"\nTraining ensemble model {model_count}/{total_models}: {result['name']}")
        
        # Load and adjust hyperparameters
        optimized_params = load_optimized_hyperparameters(model_name)
        if feature_mult != 1.0:
            optimized_params = adjust_feature_fraction_params(optimized_params, feature_mult)
        
        y_all_np = train_df.select('final_seatcount').to_numpy().flatten()
        
        if model_name == 'catboost':
            X_all = train_df.select(feature_list).to_pandas()
            X_test_final = test_df_final.select(feature_list).to_pandas()
        elif model_name == 'xgboost':
            # Handle potential infinity/NaN values for XGBoost
            X_all = train_df.select(feature_list).to_numpy()
            X_test_final = test_df_final.select(feature_list).to_numpy()
            X_all = np.nan_to_num(X_all, nan=0.0, posinf=1e6, neginf=-1e6)
            X_test_final = np.nan_to_num(X_test_final, nan=0.0, posinf=1e6, neginf=-1e6)
        else:
            X_all = train_df.select(feature_list).to_numpy()
            X_test_final = test_df_final.select(feature_list).to_numpy()
        
        model = train_final_model(model_name, X_all, y_all_np, optimized_params.copy())
        model_predictions = predict_model(model_name, model, X_test_final)
        
        ensemble_predictions.append(model_predictions)
        
        del X_all, X_test_final, y_all_np, model
        gc.collect()
    
    # Calculate ensemble predictions
    test_predictions = np.mean(ensemble_predictions, axis=0)
    print(f"\nFiltered ensemble predictions calculated from {len(ensemble_predictions)} models")
    
    # Save outputs (use features from best performing model)
    best_features = best_config['features']
    save_outputs(None, test_df_final, test_predictions, best_features, config)
    
    print(f"\n{'='*60}")
    print("FILTERED ENSEMBLE TRAINING COMPLETED")
    print(f"{'='*60}")
    print(f"Final Filtered Ensemble Val RMSE: {ensemble_val_rmse:.2f}")
    print(f"Final Filtered Ensemble Test RMSE: {ensemble_test_rmse:.2f}")
    print(f"Submission saved to: {config.get_submission_path()}")
    
    # Save experiment summary
    summary = {
        'feature_configs': FEATURE_CONFIGS,
        'models_used': MODELS_TO_RUN,
        'experiment_results': experiment_results,
        'ensemble_val_rmse': ensemble_val_rmse,
        'ensemble_test_rmse': ensemble_test_rmse,
        'best_individual': best_config,
        'total_models_in_ensemble': len(experiment_results),
        'excluded_temporal_features': ['day_of_year', 'week', 'year'],
        'excluded_problematic_features': problematic_features,
        'validation_filtering_applied': True,
        'timestamp': datetime.now().isoformat()
    }
    
    summary_file = '../outputs/v17_filtered_summary.pkl'
    with open(summary_file, 'wb') as f:
        pickle.dump(summary, f)
    
    print(f"Experiment summary saved to: {summary_file}")

if __name__ == "__main__":
    main()
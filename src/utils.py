import pandas as pd
import polars as pl
import numpy as np
from datetime import timedelta


class Config:
    """Configuration management for experiments"""
    def __init__(self, version='v1'):
        self.version = version
        self.submission_filename = f'submission_baseline_{version}.csv'
        self.feature_importance_filename = f'feature_importance_{version}.csv'
        
    def get_submission_path(self):
        return f'../outputs/{self.submission_filename}'
    
    def get_feature_importance_path(self):
        return f'../outputs/{self.feature_importance_filename}'


def load_data():
    """Load and prepare all datasets"""
    dfs = {}
    for name, path in [('train', '../data/train/train.csv'), ('test', '../data/test_8gqdJqH.csv'), ('transactions', '../data/train/transactions.csv')]:
        df = pl.read_csv(path)
        date_cols = ['doj', 'doi'] if name == 'transactions' else ['doj']
        for col in date_cols:
            if col in df.columns:
                df = df.with_columns(pl.col(col).str.strptime(pl.Date, format='%Y-%m-%d'))
        df = df.with_columns(pl.concat_str([pl.col('srcid'), pl.col('destid')], separator='_').alias('route'))
        dfs[name] = df
    return dfs['train'], dfs['test'], dfs['transactions']


def combine_train_test_data(train_df, test_df):
    """Combine train and test data with indicator column"""
    train_with_indicator = train_df.with_columns(pl.lit(0).alias('is_test'))
    test_with_indicator = test_df.with_columns([
        pl.lit(1).alias('is_test'),
        pl.lit(None).cast(pl.Float64).alias('final_seatcount')
    ]).select([
        'doj', 'srcid', 'destid', 'final_seatcount', 'route', 'is_test'
    ])
    train_with_indicator = train_with_indicator.select([
        'doj', 'srcid', 'destid', 'final_seatcount', 'route', 'is_test'
    ])
    combined_df = pl.concat([train_with_indicator, test_with_indicator])
    return combined_df


def separate_train_test_data(combined_df):
    """Separate combined data back into train and test"""
    train_df = combined_df.filter(pl.col('is_test') == 0).drop('is_test')
    test_df = combined_df.filter(pl.col('is_test') == 1).drop('is_test')
    return train_df, test_df


def create_validation_splits(feature_df):
    """Create proper validation splits"""
    df = feature_df.to_pandas()
    df['doj'] = pd.to_datetime(df['doj'])
    
    max_date = df['doj'].max()
    test_start = max_date - timedelta(days=60)
    val_start = max_date - timedelta(days=120)
    
    train_mask = df['doj'] < val_start
    val_mask = (df['doj'] >= val_start) & (df['doj'] < test_start)
    test_mask = df['doj'] >= test_start
    
    print(f"EXP-005 Split: Train={train_mask.sum()}, Val={val_mask.sum()}, Test={test_mask.sum()}")
    
    return train_mask, val_mask, test_mask, val_start


def save_outputs(model, test_df, predictions, feature_cols, config):
    """Save submission and feature importance with versioning"""
    submission = test_df.select(['route_key']).to_pandas()
    submission['final_seatcount'] = np.clip(predictions, 0, None).round().astype(int)
    submission.to_csv(config.get_submission_path(), index=False)
    
    # Handle feature importance - skip if model is None (ensemble case)
    if model is not None:
        importance_df = pd.DataFrame({'feature': feature_cols, 'importance': model.feature_importance()})
        importance_df.to_csv(config.get_feature_importance_path(), index=False)
    else:
        # For ensemble, create a placeholder feature importance file
        importance_df = pd.DataFrame({'feature': feature_cols, 'importance': [0] * len(feature_cols)})
        importance_df.to_csv(config.get_feature_importance_path(), index=False)
        print("Note: Feature importance saved as placeholder for ensemble model")

    print(f"Submission {config.version}: min={submission['final_seatcount'].min()}, max={submission['final_seatcount'].max()}, mean={submission['final_seatcount'].mean():.2f}")
    return importance_df

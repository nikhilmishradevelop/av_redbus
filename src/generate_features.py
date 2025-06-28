import polars as pl
from datetime import timedelta
import lightgbm as lgb
import holidays
import gc
import sys
import os
from utils import Config, load_data, combine_train_test_data, separate_train_test_data, create_validation_splits

def feat_eng(df, transactions_df):
    print("Starting MEMORY-OPTIMIZED feature engineering v7 (targeting 35-40k features)...")
    
    # Create mappings
    region_mapping = {region: i for i, region in enumerate(transactions_df['srcid_region'].unique().to_list())}
    tier_mapping = {tier: i for i, tier in enumerate(transactions_df['srcid_tier'].unique().to_list())}
    
    # Add encoded columns
    transactions_df = transactions_df.with_columns([
        pl.col('srcid_region').replace_strict(region_mapping, return_dtype=pl.Int32).alias('srcid_region_encoded'),
        pl.col('destid_region').replace_strict(region_mapping, return_dtype=pl.Int32).alias('destid_region_encoded'),
        pl.col('srcid_tier').replace_strict(tier_mapping, return_dtype=pl.Int32).alias('srcid_tier_encoded'),
        pl.col('destid_tier').replace_strict(tier_mapping, return_dtype=pl.Int32).alias('destid_tier_encoded')
    ])
    
    # Create lookup and join
    region_tier_lookup = transactions_df.select([
        'srcid', 'destid', 'srcid_region_encoded', 'destid_region_encoded', 
        'srcid_tier_encoded', 'destid_tier_encoded'
    ]).unique()
    
    df = df.join(region_tier_lookup, on=['srcid', 'destid'], how='left')
    
    # Clean up intermediate
    del region_tier_lookup
    gc.collect()
    
    # Holiday features
    india_holidays = holidays.India()
    
    df = df.with_columns([
        pl.col('doj').map_elements(lambda x: int(x in india_holidays), return_dtype=pl.Int64).alias('is_holiday')
    ])
    
    def get_days_to_next_holiday(date):
        for i in range(1, 366):
            next_date = date + timedelta(days=i)
            if next_date in india_holidays:
                return i
        return 365
    
    def get_days_to_prev_holiday(date):
        for i in range(1, 366):
            prev_date = date - timedelta(days=i)
            if prev_date in india_holidays:
                return i
        return 365
    
    def count_holidays_next_n_days(date, n_days):
        count = 0
        for i in range(1, n_days + 1):
            next_date = date + timedelta(days=i)
            if next_date in india_holidays:
                count += 1
        return count
    
    def count_holidays_prev_n_days(date, n_days):
        count = 0
        for i in range(1, n_days + 1):
            prev_date = date - timedelta(days=i)
            if prev_date in india_holidays:
                count += 1
        return count
    
    df = df.with_columns([
        pl.col('doj').map_elements(get_days_to_next_holiday, return_dtype=pl.Int64).alias('days_to_next_holiday'),
        pl.col('doj').map_elements(get_days_to_prev_holiday, return_dtype=pl.Int64).alias('days_to_prev_holiday'),
        pl.col('doj').map_elements(lambda x: count_holidays_next_n_days(x, 3), return_dtype=pl.Int64).alias('holidays_next_3_days'),
        pl.col('doj').map_elements(lambda x: count_holidays_prev_n_days(x, 3), return_dtype=pl.Int64).alias('holidays_prev_3_days')
    ])

    trans_filt = transactions_df.filter((pl.col('dbd') >= 15))
    trans_filt = trans_filt.with_columns([
        pl.when(pl.col('dbd') < 18).then(pl.lit('dbd_15_18'))
        .when(pl.col('dbd') < 21).then(pl.lit('dbd_18_21'))
        .when(pl.col('dbd') < 26).then(pl.lit('dbd_21_26'))
        .when(pl.col('dbd') < 31).then(pl.lit('dbd_26_31'))
        .when(pl.col('dbd') < 45).then(pl.lit('dbd_31_45'))
        .otherwise(pl.lit('dbd_45_plus'))
        .alias('dbd_bin')
    ])
    
    print("Creating transaction features...")
    # COMPREHENSIVE advanced transaction-based feature engineering
    trans_filt = trans_filt.with_columns([
        # Core differences and velocity
        pl.col('cumsum_searchcount').diff().over(['route', 'doj']).alias('searchcount_diff'),
        pl.col('cumsum_seatcount').diff().over(['route', 'doj']).alias('seatcount_diff'),
        pl.col('cumsum_seatcount').diff().diff().over(['route', 'doj']).alias('seatcount_acceleration'),
        pl.col('cumsum_searchcount').diff().diff().over(['route', 'doj']).alias('searchcount_acceleration'),
        
        # Advanced conversion and efficiency ratios
        (pl.col('cumsum_seatcount') / (1 + pl.col('cumsum_searchcount'))).alias('booking_conversion_rate'),
        (pl.col('cumsum_seatcount').pct_change().over(['route', 'doj'])).alias('seatcount_growth_rate'),
        (pl.col('cumsum_searchcount').pct_change().over(['route', 'doj'])).alias('searchcount_growth_rate'),
        (pl.col('cumsum_seatcount') / (pl.col('dbd') + 1)).alias('booking_intensity_per_day'),
        (pl.col('cumsum_searchcount') / (pl.col('dbd') + 1)).alias('search_intensity_per_day'),
        
        # Comprehensive statistical features by route-doi combinations
        pl.col('cumsum_seatcount').mean().over(['route', 'doi']).alias('seatcount_mean_by_route_doi'),
        pl.col('cumsum_seatcount').std().over(['route', 'doi']).alias('seatcount_std_by_route_doi'),
        pl.col('cumsum_seatcount').median().over(['route', 'doi']).alias('seatcount_median_by_route_doi'),
        pl.col('cumsum_searchcount').mean().over(['route', 'doi']).alias('searchcount_mean_by_route_doi'),
        pl.col('cumsum_searchcount').std().over(['route', 'doi']).alias('searchcount_std_by_route_doi'),
        
        # Extended lag features for deeper temporal patterns
        pl.col('cumsum_seatcount').shift(1).over(['route', 'doj']).alias('seatcount_lag1'),
        pl.col('cumsum_seatcount').shift(2).over(['route', 'doj']).alias('seatcount_lag2'),
        pl.col('cumsum_seatcount').shift(3).over(['route', 'doj']).alias('seatcount_lag3'),
        pl.col('cumsum_searchcount').shift(1).over(['route', 'doj']).alias('searchcount_lag1'),
        pl.col('cumsum_searchcount').shift(2).over(['route', 'doj']).alias('searchcount_lag2'),
        
        # Advanced booking pattern features (safe division)
        (pl.col('cumsum_seatcount').shift(1) / (pl.col('cumsum_seatcount') + 1)).over(['route', 'doj']).alias('seatcount_ratio_prev'),
        (pl.col('cumsum_searchcount').shift(1) / (pl.col('cumsum_searchcount') + 1)).over(['route', 'doj']).alias('searchcount_ratio_prev'),
        
    ]).with_columns([
        # Second-order derived features
        (pl.col('searchcount_diff') / (pl.col('seatcount_diff').abs() + 1)).alias('search_seat_diff_ratio'),
        (pl.col('searchcount_diff').cum_sum() / (pl.col('seatcount_diff').cum_count() + 1)).over(['route', 'doj']).alias('searchcount_diff_cummean'),
        (pl.col('seatcount_diff').cum_sum() / (pl.col('searchcount_diff').cum_count() + 1)).over(['route', 'doj']).alias('seatcount_diff_cummean'),
        
        # Comprehensive rank features for relative positioning
        pl.col('cumsum_seatcount').rank().over(['route', 'doj']).alias('seatcount_rank_in_route_doj'),
        pl.col('cumsum_seatcount').rank().over(['dbd']).alias('seatcount_rank_by_dbd'),
        pl.col('cumsum_seatcount').rank().over(['route']).alias('seatcount_rank_by_route'),
        pl.col('cumsum_searchcount').rank().over(['route', 'doj']).alias('searchcount_rank_in_route_doj'),
        pl.col('cumsum_searchcount').rank().over(['dbd']).alias('searchcount_rank_by_dbd'),
        
        # Complex momentum and efficiency scores
        (pl.col('seatcount_acceleration') * pl.col('booking_conversion_rate')).alias('momentum_score'),
        (pl.col('booking_intensity_per_day') * pl.col('booking_conversion_rate')).alias('efficiency_score'),
        (pl.col('seatcount_growth_rate') * pl.col('searchcount_growth_rate')).alias('growth_synergy_score'),
        (pl.col('seatcount_acceleration') / (pl.col('searchcount_acceleration').abs() + 1)).alias('booking_search_accel_ratio'),
        
        # Advanced rolling statistics on booking patterns
        pl.col('cumsum_seatcount').rolling_mean(window_size=3).over(['route']).alias('seatcount_rolling_mean_3'),
        pl.col('cumsum_seatcount').rolling_std(window_size=3).over(['route']).alias('seatcount_rolling_std_3'),
        pl.col('cumsum_seatcount').rolling_max(window_size=3).over(['route']).alias('seatcount_rolling_max_3'),
        pl.col('cumsum_seatcount').rolling_min(window_size=3).over(['route']).alias('seatcount_rolling_min_3'),
        pl.col('cumsum_seatcount').rolling_quantile(quantile=0.5, window_size=5).over(['route']).alias('seatcount_rolling_median_5'),
        pl.col('cumsum_seatcount').rolling_quantile(quantile=0.8, window_size=5).over(['route']).alias('seatcount_rolling_q80_5'),
        
        # Extended rolling for search patterns
        pl.col('cumsum_searchcount').rolling_mean(window_size=3).over(['route']).alias('searchcount_rolling_mean_3'),
        pl.col('cumsum_searchcount').rolling_std(window_size=3).over(['route']).alias('searchcount_rolling_std_3'),
        
        
    ]).with_columns([
        # Advanced volatility and stability features (third pass using rolling stats)
        (pl.col('seatcount_rolling_std_3') / (pl.col('seatcount_rolling_mean_3') + 1)).alias('seatcount_volatility_3'),
        (pl.col('searchcount_rolling_std_3') / (pl.col('searchcount_rolling_mean_3') + 1)).alias('searchcount_volatility_3'),
        
    ])
    
    # Clean up intermediate transaction features
    gc.collect()
    print("Creating temporal features...")
    
    date_attrs = ['weekday', 'month', 'day', 'quarter', 'week', 'ordinal_day', 'year']
    df_base = df.sort(['route', 'doj']).with_columns([
        *[getattr(pl.col('doj').dt, attr)().alias(attr if attr != 'ordinal_day' else 'day_of_year') for attr in date_attrs],
        
        # Advanced cyclical features - sine/cosine transforms for smooth seasonality
        (2 * 3.14159 * pl.col('doj').dt.day() / 31).sin().alias('day_sin'),
        (2 * 3.14159 * pl.col('doj').dt.day() / 31).cos().alias('day_cos'),
        (2 * 3.14159 * pl.col('doj').dt.month() / 12).sin().alias('month_sin'),
        (2 * 3.14159 * pl.col('doj').dt.month() / 12).cos().alias('month_cos'),
        (2 * 3.14159 * pl.col('doj').dt.weekday() / 7).sin().alias('weekday_sin'),
        (2 * 3.14159 * pl.col('doj').dt.weekday() / 7).cos().alias('weekday_cos'),
        (2 * 3.14159 * pl.col('doj').dt.week() / 52).sin().alias('week_sin'),
        (2 * 3.14159 * pl.col('doj').dt.week() / 52).cos().alias('week_cos'),
        
        # Time-distance features
        ((pl.col('doj') - pl.col('doj').min().over('route')).dt.total_days()).alias('days_since_route_start'),
        ((pl.col('doj').max().over('route') - pl.col('doj')).dt.total_days()).alias('days_until_route_end'),
        
        # Holiday interaction features  
        (pl.col('is_holiday') * pl.col('doj').dt.weekday()).alias('holiday_weekday_interaction'),
        (pl.col('is_holiday') * pl.col('doj').dt.month()).alias('holiday_month_interaction'),
        (pl.col('holidays_next_3_days') * pl.col('doj').dt.weekday()).alias('upcoming_holiday_weekday'),
        
    ]).with_columns([
        # Regional seasonal patterns - second pass using created columns
        (pl.col('srcid_region_encoded') * pl.col('month_sin')).alias('src_region_month_seasonal'),
        (pl.col('destid_region_encoded') * pl.col('month_cos')).alias('dest_region_month_seasonal'),
        (pl.col('srcid_tier_encoded') * pl.col('weekday_sin')).alias('src_tier_weekday_seasonal'),
        
        # Comprehensive polynomial time features for trend capture
        (pl.col('day_of_year') ** 2).alias('day_of_year_squared'),
        (pl.col('day_of_year') ** 3).alias('day_of_year_cubed'),
        (pl.col('week') ** 2).alias('week_squared'),
        (pl.col('month') ** 2).alias('month_squared'),
        (pl.col('days_since_route_start') ** 0.5).alias('days_since_route_start_sqrt'),
        (pl.col('days_until_route_end') ** 0.5).alias('days_until_route_end_sqrt'),
        
        # Advanced time interaction features
        (pl.col('month') * pl.col('weekday')).alias('month_weekday_interaction'),
        (pl.col('quarter') * pl.col('week')).alias('quarter_week_interaction'),
        (pl.col('is_holiday').cast(pl.Int64) * pl.col('quarter')).alias('holiday_quarter_interaction'),
        
        # Complex seasonal interaction features
        (pl.col('month_sin') * pl.col('weekday_sin')).alias('month_weekday_seasonal_interaction'),
        (pl.col('month_cos') * pl.col('weekday_cos')).alias('month_weekday_seasonal_interaction_cos'),
        (pl.col('week_sin') * pl.col('day_sin')).alias('week_day_seasonal_interaction'),
        
        # Time progression features
        (pl.col('day_of_year') / 365.0).alias('year_progress'),
        (pl.col('week') / 52.0).alias('week_progress'),
        (pl.col('days_since_route_start') / (pl.col('days_since_route_start') + pl.col('days_until_route_end') + 1)).alias('route_timeline_position'),
        
    ])
    
    # Clean up df after temporal features
    del df
    gc.collect()
    
    trans_filt = trans_filt.with_columns([
        pl.col('doj').dt.weekday().alias('weekday'),
        pl.col('doj').dt.month().alias('month'),
        pl.col('doj').dt.year().alias('year'),
        pl.col('doj').dt.day().alias('day'),
        pl.col('doj').dt.quarter().alias('quarter'),
        pl.col('doj').map_elements(lambda x: int(x in india_holidays), return_dtype=pl.Int64).alias('is_holiday'),
        pl.col('doj').dt.week().alias('week')
    ])

    print("Creating aggregations...")
    # ENHANCED AGGREGATION SETUP (v7 enhancement - moderate expansion)
    funcs = ['mean', 'std', 'min', 'max', 'median', 'count', 'last', 'first', 'sum']
    percentiles = [0.02, 0.05, 0.2, 0.8, 0.95, 0.98]
    windows = [2, 3, 5, 7, 10, 14, 21, 30, 60, 90, 180]
    shifts = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    cols = ['cumsum_seatcount', 'cumsum_searchcount', 'searchcount_diff', 'seatcount_diff',
            'seatcount_acceleration', 'searchcount_acceleration', 'booking_conversion_rate',
            'seatcount_growth_rate', 'searchcount_growth_rate', 'booking_intensity_per_day',
            'search_intensity_per_day', 'seatcount_mean_by_route_doi', 'seatcount_std_by_route_doi',
            'seatcount_median_by_route_doi', 'searchcount_mean_by_route_doi', 'searchcount_std_by_route_doi',
            'seatcount_lag1', 'seatcount_lag2', 'seatcount_lag3', 'searchcount_lag1', 'searchcount_lag2',
            'seatcount_ratio_prev', 'searchcount_ratio_prev', 'search_seat_diff_ratio', 
            'searchcount_diff_cummean', 'seatcount_diff_cummean', 'seatcount_rank_in_route_doj',
            'seatcount_rank_by_dbd', 'seatcount_rank_by_route', 'searchcount_rank_in_route_doj',
            'searchcount_rank_by_dbd', 'momentum_score', 'efficiency_score', 'growth_synergy_score',
            'booking_search_accel_ratio', 'seatcount_rolling_mean_3', 'seatcount_rolling_std_3',
            'seatcount_rolling_max_3', 'seatcount_rolling_min_3', 'seatcount_rolling_median_5',
            'seatcount_rolling_q80_5', 'searchcount_rolling_mean_3', 'searchcount_rolling_std_3',
            'seatcount_volatility_3', 'searchcount_volatility_3']

    def make_aggs(prefix, cols, funcs, percentiles):
        aggs = [getattr(pl.col(col), func)().alias(f'{prefix}_{col}_{func}') for col in cols for func in funcs if hasattr(pl.col(col), func)]
        aggs.extend([pl.col(col).quantile(p).alias(f'{prefix}_{col}_q{int(p*100)}') for col in cols for p in percentiles])
        return aggs
    
    dbd_windows = [18,21,28,35]
    
    agg_groups = [('route', ['route']),
                  ('src', ['srcid']),
                  ('dest', ['destid']),
                  ('doj', ['doj']),
                  ('trans', ['route', 'doj']),
                  ('src_dest_region', ['srcid_region_encoded', 'destid_region_encoded']),
                  ('src_dest_region_is_holiday', ['srcid_region_encoded', 'destid_region_encoded', 'is_holiday']),
                  ('src_dest_region_doj', ['srcid_region_encoded', 'destid_region_encoded', 'doj']),
                  ('src_dest_region_year_week', ['srcid_region_encoded', 'destid_region_encoded', 'year', 'week']),
                  ('src_dest_region_year_month', ['srcid_region_encoded', 'destid_region_encoded', 'year', 'month']),
                  ('route_year_week', ['route', 'year', 'week']),
                  ('src_doj', ['srcid', 'doj']),
                  ('dest_doj', ['destid', 'doj']),
                  ('route_month', ['route', 'month']), 
                  ('route_weekday', ['route', 'weekday']),
                  ('route_is_holiday', ['route', 'is_holiday']),
                  ('source_month', ['srcid', 'month']),
                  ('dest_month', ['destid', 'month']),
                  ('source_weekday', ['srcid', 'weekday']),
                  ('dest_weekday', ['destid', 'weekday']),
                  ('route_year_month', ['route', 'year', 'month'])]

    for i, (prefix, group_cols) in enumerate(agg_groups):
        print(f"Processing aggregation group {i+1}/{len(agg_groups)}: {prefix}")
        
        if prefix == 'trans':
            agg_df = trans_filt.group_by(group_cols).agg([
                *make_aggs(prefix, cols, funcs, percentiles),
                *[(pl.col('dbd') <= w).sum().alias(f'bookings_within_{w}d') for w in dbd_windows],
                *[(pl.col('dbd') >= w).sum().alias(f'bookings_after_{w}d') for w in dbd_windows],
            ]).sort(['route', 'doj']).with_columns([
                *[getattr(pl.col(f'trans_{col}_{func}'), f'rolling_{rfunc}')(w).over('route').alias(f'{col}_{func}_{w}d_{rfunc}') for col in cols for func in funcs[:5] for w in windows for rfunc in ['mean', 'std', 'max', 'min'] if hasattr(pl.col(f'trans_{col}_{func}'), f'rolling_{rfunc}')],
                *[pl.col(f'trans_{col}_{func}').shift(s).over('route').alias(f'{col}_{func}_lag{s}') for col in cols for func in funcs[:5] for s in shifts],
                *[pl.col(f'trans_{col}_{func}').pct_change(s).over('route').alias(f'{col}_{func}_pct{s}') for col in cols for func in funcs[:3] for s in shifts],
            ])
        else:
            agg_df = trans_filt.group_by(group_cols).agg([
                *make_aggs(prefix, cols, funcs, percentiles),
                pl.col('route').n_unique().alias(f'{prefix}_unique_routes'),
            ])
        
        df_base = df_base.join(agg_df, on=group_cols, how='left')
        
        # Clean up intermediate aggregation
        del agg_df
        gc.collect()
            
    route_dbd_agg = trans_filt.group_by(['route', 'dbd_bin']).agg([
        *make_aggs('route_dbd', cols, funcs, percentiles),
    ])
    
    dbd_labels = ['dbd_15_18', 'dbd_18_21', 'dbd_21_26', 'dbd_26_31', 'dbd_31_45', 'dbd_45_plus']
    for bin_name in dbd_labels:
        bin_agg = route_dbd_agg.filter(pl.col('dbd_bin') == bin_name).drop('dbd_bin')
        bin_agg = bin_agg.rename({col: f'{col}_{bin_name}' for col in bin_agg.columns if col != 'route'})
        df_base = df_base.join(bin_agg, on='route', how='left')
        del bin_agg
        gc.collect()
    
    # Clean up large intermediate dataframes
    del trans_filt, route_dbd_agg
    gc.collect()
    
    print(f"Feature engineering completed. Total columns: {len(df_base.columns)}")
    return df_base

def main():
    print("Loading data...")
    train_df, test_df, transactions_df = load_data()
    df = combine_train_test_data(train_df, test_df)
    
    # Clean up original dataframes
    del train_df, test_df
    gc.collect()
    
    print("Starting feature engineering...")
    df = feat_eng(df, transactions_df)
    
    # Clean up transactions_df after feature engineering
    del transactions_df
    gc.collect()
    
    train_df, test_df = separate_train_test_data(df)
    del df  # Clean up combined df
    gc.collect()
    
    train_mask, val_mask, _, _ = create_validation_splits(train_df)
    
    exclude_cols = ['doj', 'route', 'final_seatcount', 'srcid', 'destid', 'year']
    feature_cols = [col for col in train_df.columns 
                   if col not in exclude_cols and 
                   str(train_df[col].dtype) not in ['Date', 'Datetime']]
    
    print("Selected {} features initially".format(len(feature_cols)))
    
    # Extract target first
    y_np = train_df.select('final_seatcount').to_numpy().flatten()
    y_train, y_val = y_np[train_mask], y_np[val_mask]
    
    # Keep train_df in polars format for memory efficiency
    X_train_pl = train_df.filter(pl.Series(train_mask))
    X_val_pl = train_df.filter(pl.Series(val_mask))  
    
    # Clean up intermediate arrays
    del y_np, train_mask, val_mask
    gc.collect()
    
    # Run feature selection to get importance
    print("\n" + "="*60)
    print("STEP 1: FEATURE SELECTION WITH ALL FEATURES")
    print("Using {} features for feature selection".format(len(feature_cols)))
    print("="*60)
    
    feature_selection_params = {
        'objective': 'regression',
        'metric': 'rmse',
        'verbosity': -1,
        'random_state': 42,
        'learning_rate': 0.03,
        'max_bin': 25,
        'max_depth': -1,
        'min_data_in_leaf': 30,
        'feature_fraction': 0.01,
        'lambda_l1': 10,
        'lambda_l2': 10,
        'min_gain_to_split': 0,
        'num_leaves': 105
    }
    
    # Convert to numpy only for LightGBM training
    X_train_selection = X_train_pl.select(feature_cols).to_numpy()
    X_val_selection = X_val_pl.select(feature_cols).to_numpy()
    
    # Train feature selection model (validation only, no test)
    feature_selection_model = lgb.train(
        feature_selection_params,
        lgb.Dataset(X_train_selection, label=y_train),
        valid_sets=[lgb.Dataset(X_val_selection, label=y_val)],
        num_boost_round=450,
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(50)]
    )
    
    # Clean up selection arrays
    del X_train_selection, X_val_selection
    gc.collect()
    
    # Get feature importance and save to CSV
    feature_importance = feature_selection_model.feature_importance(importance_type='gain')
    importance_pl = pl.DataFrame({
        'feature': feature_cols,
        'importance': feature_importance
    }).sort('importance', descending=True)
    
    # Save feature importance to CSV for future use
    os.makedirs('../outputs', exist_ok=True)
    importance_df = importance_pl.to_pandas()
    feature_importance_file = '../outputs/feature_importance.csv'
    importance_df.to_csv(feature_importance_file, index=False)
    print("\nSaved feature importance to: {}".format(feature_importance_file))
    
    # Extract top 6000 features
    top_6k_features = importance_pl.head(6000)['feature'].to_list()
    
    print("Selected top 6000 features based on importance")
    
    # Clean up
    del feature_selection_model, feature_importance, importance_pl, importance_df
    gc.collect()
    
    # Create dataframe with top 6000 features + required columns for submissions
    required_cols = ['doj', 'route', 'final_seatcount', 'srcid', 'destid']
    
    # Combine training and test data with top features
    print("Creating optimized dataset with top 6000 features...")
    train_df_optimized = train_df.select(required_cols + top_6k_features)
    test_df_optimized = test_df.select([col for col in required_cols + top_6k_features if col in test_df.columns])
    
    # Combine for full dataset
    df_optimized = pl.concat([train_df_optimized, test_df_optimized], how='diagonal')
    
    print("Optimized dataset shape: {}".format(df_optimized.shape))
    print("Columns: {}".format(len(df_optimized.columns)))
    
    # Save as IPC format
    output_file = '../outputs/optimized_dataset_top6k_features.ipc'
    df_optimized.write_ipc(output_file)
    print("Saved optimized dataset to: {}".format(output_file))
    
    # Save top 6000 feature names as well
    top_6k_df = pl.DataFrame({'feature': top_6k_features})
    top_6k_df.write_csv('../outputs/top_6000_features.csv')
    print("Saved top 6000 feature names to: ../outputs/top_6000_features.csv")
    
    print("\nFeature extraction and optimization completed!")

if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
Ensemble V17 Filtered model with TabDPT model
V17 weight: 0.75, TabDPT weight: 0.25
"""

import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr, spearmanr

def create_ensemble_submission():
    """Create weighted ensemble of V17 and TabDPT models"""
    
    print("=== V17 + TabDPT ENSEMBLE CREATOR ===")
    print(f"Timestamp: {datetime.now()}")
    print("Weights: V17=0.75, TabDPT=0.25")
    
    # Load submission files
    print("\nLoading submission files...")
    v17_file = "../outputs/submission_baseline_v17_filtered.csv"
    tabdpt_ensemble_file = "../outputs/submission_baseline_tabdpt_model.csv"
    
    try:
        v17_sub = pd.read_csv(v17_file)
        tabdpt_ensemble_sub = pd.read_csv(tabdpt_ensemble_file)
        print(f"V17 shape: {v17_sub.shape}")
        print(f"TabDPT shape: {tabdpt_ensemble_sub.shape}")
    except FileNotFoundError as e:
        print(f"Error loading files: {e}")
        return None
    
    # Verify route_key columns exist
    if 'route_key' not in v17_sub.columns or 'route_key' not in tabdpt_ensemble_sub.columns:
        print("Error: route_key column missing from one or both files")
        return None
    
    # Merge submissions on route_key
    print("\nMerging submissions...")
    merged = v17_sub.merge(tabdpt_ensemble_sub, on='route_key', suffixes=('_v17', '_tabdpt'))
    print(f"Merged shape: {merged.shape}")
    
    if len(merged) != len(v17_sub):
        print(f"Warning: {len(v17_sub) - len(merged)} routes lost in merge")
    
    # Extract predictions
    v17_predictions = merged['final_seatcount_v17']
    tabdpt_predictions = merged['final_seatcount_tabdpt']
    
    # Create weighted ensemble
    final_ensemble_predictions = 0.75 * v17_predictions + 0.25 * tabdpt_predictions
    
    print("\nModel Statistics:")
    print(f"V17 mean: {v17_predictions.mean():.2f}")
    print(f"TabDPT mean: {tabdpt_predictions.mean():.2f}")
    print(f"Final Ensemble mean: {final_ensemble_predictions.mean():.2f}")
    
    # Calculate correlation between models
    corr_pearson, _ = pearsonr(v17_predictions, tabdpt_predictions)
    corr_spearman, _ = spearmanr(v17_predictions, tabdpt_predictions)
    rmse_between = np.sqrt(mean_squared_error(v17_predictions, tabdpt_predictions))
    
    print(f"\nModel Comparison:")
    print(f"RMSE between V17 and TabDPT: {rmse_between:.2f}")
    print(f"Pearson correlation: {corr_pearson:.4f}")
    print(f"Spearman correlation: {corr_spearman:.4f}")
    
    # Create final submission dataframe
    final_ensemble_submission = pd.DataFrame({
        'route_key': merged['route_key'],
        'final_seatcount': final_ensemble_predictions.round().astype(int)
    })
    
    # Ensure non-negative predictions
    final_ensemble_submission['final_seatcount'] = final_ensemble_submission['final_seatcount'].clip(lower=0)
    
    print(f"\nFinal Ensemble Statistics:")
    print(f"Min prediction: {final_ensemble_submission['final_seatcount'].min()}")
    print(f"Max prediction: {final_ensemble_submission['final_seatcount'].max()}")
    print(f"Mean prediction: {final_ensemble_submission['final_seatcount'].mean():.2f}")
    print(f"Std prediction: {final_ensemble_submission['final_seatcount'].std():.2f}")
    
    # Save ensemble submission
    output_file = "../outputs/submission_ensemble_v17_tabdpt_75_25.csv"
    final_ensemble_submission.to_csv(output_file, index=False)
    print(f"\nFinal ensemble submission saved: {output_file}")
    
    # Show sample predictions
    print(f"\nSample predictions (first 10 routes):")
    sample_df = merged[['route_key', 'final_seatcount_v17', 'final_seatcount_tabdpt']].head(10)
    sample_df['final_ensemble_prediction'] = final_ensemble_predictions.head(10).round().astype(int)
    print(sample_df.to_string(index=False))
    
    # Calculate individual model weights contribution
    v17_contribution = (0.75 * v17_predictions).mean()
    tabdpt_contribution = (0.25 * tabdpt_predictions).mean()
    
    print(f"\nWeight Contributions:")
    print(f"V17 contribution (75%): {v17_contribution:.2f}")
    print(f"TabDPT contribution (25%): {tabdpt_contribution:.2f}")
    print(f"Total final ensemble mean: {v17_contribution + tabdpt_contribution:.2f}")
    
    return final_ensemble_submission, output_file

def main():
    """Main execution function"""
    result = create_ensemble_submission()
    
    if result is not None:
        ensemble_df, output_file = result
        print(f"\n{'='*50}")
        print("FINAL ENSEMBLE CREATION COMPLETED SUCCESSFULLY")
        print(f"Final submission: {output_file}")
        print(f"Total routes: {len(ensemble_df)}")
        print(f"Architecture: V17 (75%) + TabDPT (25%)")
        print(f"{'='*50}")
    else:
        print("\nEnsemble creation failed!")

if __name__ == "__main__":
    main()
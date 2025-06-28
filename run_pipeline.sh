#!/bin/bash

# RedBus Competition Pipeline Execution Script
# Run this after setup is complete (when venv and dependencies are already installed)

set -e  # Exit on any error

echo "======================================"
echo "RedBus Competition Pipeline Execution"
echo "======================================"

# Configuration
VENV_NAME="redbus_env"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Step 1: Check if virtual environment exists
if [ ! -d "$VENV_NAME" ]; then
    print_error "Virtual environment '$VENV_NAME' not found!"
    print_error "Please run './setup_and_run.sh' first to set up the environment."
    exit 1
fi

# Step 2: Activate virtual environment
print_status "Activating virtual environment: $VENV_NAME"
source "$VENV_NAME/bin/activate"

# Verify activation
if [ "$VIRTUAL_ENV" == "" ]; then
    print_error "Failed to activate virtual environment"
    exit 1
fi

print_status "Virtual environment activated: $VIRTUAL_ENV"

# Step 3: Verify data files exist
print_status "Checking data files..."
if [ ! -f "data/train/train.csv" ] || [ ! -f "data/train/transactions.csv" ] || [ ! -f "data/test_8gqdJqH.csv" ]; then
    print_error "Required data files not found in data/ directory:"
    echo "  - data/train/train.csv"
    echo "  - data/train/transactions.csv" 
    echo "  - data/test_8gqdJqH.csv"
    print_warning "Please place the competition data files in the correct directories before running."
    exit 1
fi

print_status "All data files found"

# Step 4: Check if TabDPT is installed
print_status "Checking TabDPT installation..."
python -c "import tabdpt; print('TabDPT version:', tabdpt.__version__ if hasattr(tabdpt, '__version__') else 'installed')" 2>/dev/null
if [ $? -ne 0 ]; then
    print_error "TabDPT not found! Please run './setup_and_run.sh' first."
    exit 1
fi

print_status "TabDPT is installed and available"

# Step 5: Change to src directory for pipeline execution
print_status "Changing to src directory for pipeline execution..."
cd src

# Step 6: Run the complete pipeline
print_status "Starting the complete pipeline..."
echo ""
echo "Pipeline execution order:"
echo "1. Feature generation (generate_features.py)"
echo "2. GBM models training (gbm_model.py)"
echo "3. TabDPT model training (tabdpt_model.py)"
echo "4. Ensemble creation (ensemble.py)"
echo ""

# Create outputs directory if it doesn't exist
mkdir -p ../outputs

# Step 6.1: Feature Generation
print_status "Step 1/4: Running feature generation..."
python generate_features.py
if [ $? -eq 0 ]; then
    print_status "Feature generation completed successfully"
else
    print_error "Feature generation failed"
    exit 1
fi

# Step 6.2: GBM Models
print_status "Step 2/4: Training GBM models (LightGBM, XGBoost, CatBoost)..."
python gbm_model.py
if [ $? -eq 0 ]; then
    print_status "GBM models training completed successfully"
else
    print_error "GBM models training failed"
    exit 1
fi

# Step 6.3: TabDPT Model
print_status "Step 3/4: Training TabDPT model..."
python tabdpt_model.py
if [ $? -eq 0 ]; then
    print_status "TabDPT model training completed successfully"
else
    print_error "TabDPT model training failed"
    exit 1
fi

# Step 6.4: Ensemble
print_status "Step 4/4: Creating ensemble submission..."
python ensemble.py
if [ $? -eq 0 ]; then
    print_status "Ensemble creation completed successfully"
else
    print_error "Ensemble creation failed"
    exit 1
fi

# Step 7: Display results
print_status "Pipeline completed successfully!"
echo ""
echo "Generated files in outputs/:"
ls -la ../outputs/submission_*.csv 2>/dev/null || print_warning "No submission files found"

echo ""
echo "======================================"
echo "COMPETITION SUBMISSION READY"
echo "======================================"
print_status "Final submission file: outputs/submission_ensemble_v17_tabdpt_75_25.csv"
echo ""
print_status "Pipeline execution completed!"
print_status "You can run this script again anytime without reinstalling dependencies."
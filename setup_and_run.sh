#!/bin/bash

# RedBus Competition Setup and Execution Script
# This script creates a virtual environment, installs dependencies, and runs the complete pipeline

set -e  # Exit on any error

echo "======================================"
echo "RedBus Competition Setup and Execution"
echo "======================================"

# Configuration
VENV_NAME="redbus_env"
PYTHON_VERSION="python3"

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

# Step 1: Check Python installation
print_status "Checking Python installation..."
if command -v $PYTHON_VERSION &> /dev/null; then
    PYTHON_VER=$($PYTHON_VERSION --version)
    print_status "Found $PYTHON_VER"
else
    print_error "Python3 is not installed. Please install Python 3.8+ first."
    exit 1
fi

# Step 2: Create virtual environment
print_status "Creating virtual environment: $VENV_NAME"
if [ -d "$VENV_NAME" ]; then
    print_warning "Virtual environment already exists. Removing it..."
    rm -rf "$VENV_NAME"
fi

$PYTHON_VERSION -m venv "$VENV_NAME"
print_status "Virtual environment created successfully"

# Step 3: Activate virtual environment
print_status "Activating virtual environment..."
source "$VENV_NAME/bin/activate"

# Verify activation
if [ "$VIRTUAL_ENV" != "" ]; then
    print_status "Virtual environment activated: $VIRTUAL_ENV"
else
    print_error "Failed to activate virtual environment"
    exit 1
fi

# Step 4: Upgrade pip
print_status "Upgrading pip..."
pip install --upgrade pip

# Step 5: Install requirements
print_status "Installing requirements..."
if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt
    print_status "Requirements installed successfully"
else
    print_error "requirements.txt not found"
    exit 1
fi

# Step 6: Install TabDPT (special handling)
print_status "Installing TabDPT..."
print_warning "This may take a while as it downloads model weights..."

# Check if git and git-lfs are available
if ! command -v git &> /dev/null; then
    print_error "Git is not installed. Please install git first."
    exit 1
fi

if ! command -v git-lfs &> /dev/null; then
    print_warning "Git LFS not found. Installing git-lfs..."
    # Try to install git-lfs (works on most systems)
    if command -v apt-get &> /dev/null; then
        sudo apt-get update && sudo apt-get install -y git-lfs
    elif command -v yum &> /dev/null; then
        sudo yum install -y git-lfs
    elif command -v brew &> /dev/null; then
        brew install git-lfs
    else
        print_error "Could not install git-lfs automatically. Please install it manually."
        print_error "On Ubuntu/Debian: sudo apt-get install git-lfs"
        print_error "On RHEL/CentOS: sudo yum install git-lfs"
        print_error "On macOS: brew install git-lfs"
        exit 1
    fi
fi

# Initialize and update git submodules (includes TabDPT)
print_status "Initializing git submodules..."
git submodule init
git submodule update

# Verify TabDPT directory exists and has content
if [ ! -d "TabDPT" ] || [ ! -f "TabDPT/pyproject.toml" ]; then
    print_error "TabDPT submodule not properly initialized"
    print_status "Attempting manual clone as fallback..."
    rm -rf TabDPT
    git clone https://github.com/layer6ai-labs/TabDPT.git
    if [ ! -f "TabDPT/pyproject.toml" ]; then
        print_error "Failed to clone TabDPT repository"
        exit 1
    fi
else
    print_status "TabDPT submodule initialized successfully"
fi

# Install TabDPT
print_status "Installing TabDPT in editable mode..."
cd TabDPT

# Pull the latest model weights using git LFS
print_status "Pulling model weights with git LFS..."
git lfs pull

# Install TabDPT in editable mode
pip install -e .

if [ $? -eq 0 ]; then
    print_status "TabDPT installed successfully"
else
    print_error "TabDPT installation failed"
    exit 1
fi

# Return to main directory
cd ..

# Step 7: Verify data files exist
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

# Step 7.5: Change to src directory for pipeline execution
print_status "Changing to src directory for pipeline execution..."
cd src

# Step 8: Run the complete pipeline
print_status "Starting the complete pipeline..."
echo ""
echo "Pipeline execution order:"
echo '1. Feature generation (generate_features.py)'
echo '2. GBM models training (gbm_model.py)'
echo '3. TabDPT model training (tabdpt_model.py)'
echo '4. Ensemble creation (ensemble.py)'
echo ""

# Create outputs directory if it doesn't exist
mkdir -p ../outputs

# Step 8.1: Feature Generation
print_status "Step 1/4: Running feature generation..."
python generate_features.py
if [ $? -eq 0 ]; then
    print_status "Feature generation completed successfully"
else
    print_error "Feature generation failed"
    exit 1
fi

# Step 8.2: GBM Models
print_status "Step 2/4: Training GBM models (LightGBM, XGBoost, CatBoost)..."
python gbm_model.py
if [ $? -eq 0 ]; then
    print_status "GBM models training completed successfully"
else
    print_error "GBM models training failed"
    exit 1
fi

# Step 8.3: TabDPT Model
print_status "Step 3/4: Training TabDPT model..."
python tabdpt_model.py
if [ $? -eq 0 ]; then
    print_status "TabDPT model training completed successfully"
else
    print_error "TabDPT model training failed"
    exit 1
fi

# Step 8.4: Ensemble
print_status "Step 4/4: Creating ensemble submission..."
python ensemble.py
if [ $? -eq 0 ]; then
    print_status "Ensemble creation completed successfully"
else
    print_error "Ensemble creation failed"
    exit 1
fi

# Step 9: Display results
print_status "Pipeline completed successfully!"
echo ""
echo "Generated files in outputs/:"
ls -la ../outputs/submission_*.csv 2>/dev/null || print_warning "No submission files found"

echo ""
echo "======================================"
echo "COMPETITION SUBMISSION READY"
echo "======================================"
print_status "Final submission file: outputs/submission_ensemble_v17_tabdpt_75_25.csv"
print_status "Virtual environment: $VENV_NAME (keep this for future runs)"
echo ""
print_status "To run individual components later:"
echo "  source $VENV_NAME/bin/activate"
echo "  cd src"
echo "  python generate_features.py  # Generate features"
echo "  python gbm_model.py          # Train GBM models"
echo "  python tabdpt_model.py       # Train TabDPT model"
echo "  python ensemble.py           # Create ensemble"

print_status "Setup and pipeline execution completed!"
echo ""
print_status "NEXT TIME: You can skip setup and just run the pipeline with:"
print_status "  ./run_pipeline.sh"
#!/bin/bash
# Setup script for TDA Backend development environment

set -e

echo "🚀 Setting up TDA Backend development environment..."

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check Python version
print_status "Checking Python version..."
python_version=$(python3 --version 2>&1 | awk '{print $2}')
required_version="3.11"

if python3 -c "import sys; sys.exit(0 if sys.version_info >= (3, 11) else 1)"; then
    print_success "Python $python_version is compatible (>= $required_version)"
else
    print_error "Python $python_version is not compatible. Required: >= $required_version"
    exit 1
fi

# Create virtual environment
print_status "Creating virtual environment..."
if [ ! -d ".venv" ]; then
    python3 -m venv .venv
    print_success "Virtual environment created"
else
    print_warning "Virtual environment already exists"
fi

# Activate virtual environment
print_status "Activating virtual environment..."
source .venv/bin/activate

# Upgrade pip
print_status "Upgrading pip..."
pip install --upgrade pip

# Install dependencies
print_status "Installing dependencies..."
if [ -f "pyproject.toml" ]; then
    pip install -e .[dev]
    print_success "Dependencies installed from pyproject.toml"
else
    print_error "pyproject.toml not found"
    exit 1
fi

# Create .env file if it doesn't exist
print_status "Setting up environment configuration..."
if [ ! -f ".env" ]; then
    cp .env.example .env
    print_success "Created .env file from .env.example"
    print_warning "Please edit .env file with your configuration"
else
    print_warning ".env file already exists"
fi

# Create required directories
print_status "Creating required directories..."
directories=(
    "logs"
    "tmp"
    "uploads"
    "/tmp/tda-uploads"
    "/tmp/prometheus_multiproc"
)

for dir in "${directories[@]}"; do
    if [ ! -d "$dir" ]; then
        mkdir -p "$dir"
        print_success "Created directory: $dir"
    fi
done

# Setup pre-commit hooks (if available)
if command -v pre-commit &> /dev/null; then
    print_status "Setting up pre-commit hooks..."
    pre-commit install
    print_success "Pre-commit hooks installed"
else
    print_warning "pre-commit not found. Install with: pip install pre-commit"
fi

# Check if Docker is available
if command -v docker &> /dev/null; then
    print_success "Docker is available for running external services"
    
    if command -v docker-compose &> /dev/null; then
        print_success "Docker Compose is available"
    else
        print_warning "Docker Compose not found. Some services may not be available."
    fi
else
    print_warning "Docker not found. External services will need to be installed manually."
fi

# Display next steps
echo
print_success "🎉 Setup completed successfully!"
echo
echo -e "${BLUE}Next steps:${NC}"
echo "1. Activate the virtual environment:"
echo "   source .venv/bin/activate"
echo
echo "2. Edit the .env file with your configuration:"
echo "   nano .env"
echo
echo "3. Start external services (PostgreSQL, Redis, Kafka, Flink):"
echo "   docker-compose up -d"
echo
echo "4. Run database migrations:"
echo "   tda-backend migrate"
echo
echo "5. Start the development server:"
echo "   tda-backend server --reload"
echo
echo "6. View API documentation at:"
echo "   http://localhost:8000/docs"
echo
echo -e "${YELLOW}For more information, see the README.md file.${NC}"
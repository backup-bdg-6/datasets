#!/bin/bash

# Deployment script for AI model

set -e

# Configuration
MODEL_DIR="outputs/checkpoints"
TOKENIZER_DIR="outputs/tokenizer"
CONFIG_DIR="outputs/flask_deployment"

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Print header
echo -e "${GREEN}=========================================${NC}"
echo -e "${GREEN}   AI Model Deployment Script           ${NC}"
echo -e "${GREEN}=========================================${NC}"

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo -e "${RED}Error: Docker is not installed.${NC}"
    echo "Please install Docker and try again."
    exit 1
fi

# Check if Docker Compose is installed
if ! command -v docker-compose &> /dev/null; then
    echo -e "${RED}Error: Docker Compose is not installed.${NC}"
    echo "Please install Docker Compose and try again."
    exit 1
fi

# Check if model files exist
if [ ! -f "${MODEL_DIR}/model_final.pt" ]; then
    echo -e "${RED}Error: Model file not found at ${MODEL_DIR}/model_final.pt${NC}"
    echo "Please train the model first or specify the correct path."
    exit 1
fi

if [ ! -f "${TOKENIZER_DIR}/tokenizer.json" ]; then
    echo -e "${RED}Error: Tokenizer file not found at ${TOKENIZER_DIR}/tokenizer.json${NC}"
    echo "Please train the model first or specify the correct path."
    exit 1
fi

# Create config directory if it doesn't exist
mkdir -p ${CONFIG_DIR}

# Convert YAML config to JSON if it doesn't exist
if [ ! -f "${CONFIG_DIR}/config.json" ]; then
    echo -e "${YELLOW}Converting config.yaml to config.json...${NC}"
    python -c "
import yaml
import json
import os

config_yaml = 'configs/config.yaml'
config_json = '${CONFIG_DIR}/config.json'

if os.path.exists(config_yaml):
    with open(config_yaml, 'r') as f:
        config = yaml.safe_load(f)
    
    with open(config_json, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f'Config converted and saved to {config_json}')
else:
    print(f'Error: {config_yaml} not found')
    exit(1)
"
fi

# Build and start the Docker containers
echo -e "${GREEN}Building and starting Docker containers...${NC}"
docker-compose up --build -d

# Check if containers are running
if [ "$(docker-compose ps -q | wc -l)" -eq 2 ]; then
    echo -e "${GREEN}Deployment successful!${NC}"
    echo -e "API is available at http://localhost/api/"
    echo -e "Health check: http://localhost/health"
else
    echo -e "${RED}Deployment failed. Please check the logs:${NC}"
    docker-compose logs
    exit 1
fi

echo -e "${GREEN}=========================================${NC}"
echo -e "${YELLOW}To stop the deployment:${NC} docker-compose down"
echo -e "${YELLOW}To view logs:${NC} docker-compose logs -f"
echo -e "${GREEN}=========================================${NC}"
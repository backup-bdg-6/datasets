import os
import shutil
import psutil
import subprocess
from huggingface_hub import snapshot_download, HfApi, login
from huggingface_hub.utils import HFValidationError

# Login to Hugging Face with the provided token from environment variable
hf_token = os.environ.get("HF_TOKEN")
if not hf_token:
    print("Warning: HF_TOKEN environment variable not set. Some datasets may not be accessible.")

try:
    login(token=hf_token)
    print("Successfully logged in to Hugging Face.")
except Exception as e:
    print(f"Failed to login to Hugging Face: {e}")

# List of datasets to download
datasets = [
    "PrimeIntellect/verifiable-coding-problems",
    "DeepNLP/Coding-Agent-GitHub-2025-Feb",
    "aidando73/llama-coding-agent-evals",
    "HumanLLMs/Human-Like-DPO-Dataset",
    "HuggingFaceH4/ultrachat_200k",
    "Anthropic/hh-rlhf",
    "xAI/TruthfulQA",
    "Salesforce/dialogstudio",
    "bigcode/the-stack",
    "allenai/tool-augmented-dialogues",
    "google-research/toolbench",
    "codeparrot/github-code",
    "luahub/lua-code-dataset",
    "roblox/luau-code",
    "swift-code/swift-repos",
    "HuggingFaceH4/codeparrot-ds",
    "nuprl/MultiPL-E",
    "codeparrot/apps",
    "HuggingFaceH4/python-codes-25k",
    "HuggingFaceH4/instruction-dataset",
    "allenai/dolma",
    "openwebtext/openwebtext"
]

# Function to get available disk space in MB
def get_available_disk_space():
    disk = psutil.disk_usage('.')
    return disk.free / (1024 * 1024)  # Convert to MB

# Function to estimate dataset size
def estimate_dataset_size(dataset_id):
    try:
        api = HfApi()
        dataset_info = api.dataset_info(dataset_id)
        size_bytes = dataset_info.size
        return size_bytes / (1024 * 1024)  # Convert to MB
    except Exception as e:
        print(f"Could not estimate size for {dataset_id}: {e}")
        return float('inf')  # Return infinity if size cannot be determined

# Function to download a dataset
def download_dataset(dataset_id):
    # Create a clean directory name
    dir_name = dataset_id.replace('/', '_')
    local_dir = f"datasets/{dir_name}"
    
    # Skip if already downloaded
    if os.path.exists(local_dir) and os.listdir(local_dir):
        print(f"Dataset {dataset_id} already exists. Skipping...")
        return True
    
    # Estimate dataset size
    estimated_size = estimate_dataset_size(dataset_id)
    print(f"Estimated size of {dataset_id}: {estimated_size:.2f} MB")
    
    # Check available disk space
    available_space = get_available_disk_space()
    print(f"Available disk space: {available_space:.2f} MB")
    
    # Skip if not enough space
    if estimated_size > available_space:
        print(f"Skipping {dataset_id}: Not enough disk space (required: {estimated_size:.2f} MB, available: {available_space:.2f} MB)")
        return False
    
    # Download the dataset
    print(f"Downloading {dataset_id}...")
    try:
        os.makedirs(local_dir, exist_ok=True)
        snapshot_download(
            repo_id=dataset_id,
            repo_type="dataset",
            local_dir=local_dir
        )
        print(f"Successfully downloaded {dataset_id}")
        
        # Clean up cache to save space
        cache_dir = os.path.join(local_dir, ".cache")
        if os.path.exists(cache_dir):
            print(f"Cleaning up {cache_dir}...")
            shutil.rmtree(cache_dir)
        
        return True
    except Exception as e:
        print(f"Failed to download {dataset_id}: {e}. Skipping...")
        # Clean up the directory if download failed
        if os.path.exists(local_dir):
            shutil.rmtree(local_dir)
        return False

# Function to add dataset to Git
def add_to_git(dataset_id):
    dir_name = dataset_id.replace('/', '_')
    local_dir = f"datasets/{dir_name}"
    
    print(f"Adding {local_dir} to Git...")
    try:
        subprocess.run(["git", "add", local_dir], check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Failed to add {local_dir} to Git: {e}")
        return False

# Function to commit dataset
def commit_dataset(dataset_id):
    print(f"Committing {dataset_id}...")
    try:
        subprocess.run(["git", "commit", "-m", f"Add dataset: {dataset_id}"], check=True)
        print(f"Successfully committed {dataset_id}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Failed to commit {dataset_id}: {e}")
        return False

# Function to clean up dataset to free up space
def cleanup_dataset(dataset_id):
    dir_name = dataset_id.replace('/', '_')
    local_dir = f"datasets/{dir_name}"
    
    print(f"Cleaning up {local_dir} to free up space...")
    try:
        # Remove the directory
        shutil.rmtree(local_dir)
        return True
    except Exception as e:
        print(f"Failed to clean up {local_dir}: {e}")
        return False

# Download and commit each dataset
for dataset_id in datasets:
    # Download the dataset
    if download_dataset(dataset_id):
        # Add to Git
        if add_to_git(dataset_id):
            # Commit the dataset
            commit_dataset(dataset_id)
        
        # Clean up to free space
        cleanup_dataset(dataset_id)

print("Download and commit complete!")
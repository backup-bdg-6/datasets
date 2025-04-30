from huggingface_hub import snapshot_download
import os

# List of datasets to download
datasets = [
    "nvidia/OpenCodeReasoning",
    "open-thoughts/OpenThoughts2-1M",
    "Anthropic/values-in-the-wild",
    "PrimeIntellect/verifiable-coding-problems",
    "DeepNLP/Coding-Agent-GitHub-2025-Feb",
    "aidando73/llama-coding-agent-evals",
    "openai/openai_humaneval",
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

# Directory to store datasets
base_dir = "datasets"
os.makedirs(base_dir, exist_ok=True)

# Download each dataset
for dataset in datasets:
    try:
        print(f"Downloading {dataset}...")
        snapshot_download(
            repo_id=dataset,
            repo_type="dataset",
            local_dir=f"{base_dir}/{dataset.replace('/', '_')}",
            allow_patterns=["*.jsonl", "*.csv", "*.parquet", "*.txt"],
            ignore_patterns=["*.md", "*.ipynb"]
        )
        print(f"Successfully downloaded {dataset}")
    except Exception as e:
        print(f"Failed to download {dataset}: {e}")

print("Download complete!")

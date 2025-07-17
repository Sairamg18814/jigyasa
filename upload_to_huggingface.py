#!/usr/bin/env python3
"""
Upload JIGYASA to Hugging Face Hub
"""

from huggingface_hub import HfApi, create_repo, upload_folder
import os

def upload_to_huggingface():
    # Initialize API
    api = HfApi()
    
    # Get token from environment
    token = os.getenv("HUGGINGFACE_TOKEN")
    if not token:
        print("❌ Please set HUGGINGFACE_TOKEN environment variable")
        return False
    
    # Create repository
    repo_id = "Sairamg18814/jigyasa-agi"
    try:
        create_repo(repo_id, token=token, repo_type="model", exist_ok=True)
        print(f"✅ Repository created/exists: {repo_id}")
    except Exception as e:
        print(f"❌ Error creating repository: {e}")
        return False
    
    # Upload files
    try:
        upload_folder(
            folder_path="huggingface_package",
            repo_id=repo_id,
            token=token,
            commit_message="Upload JIGYASA AGI model v1.0.0"
        )
        print(f"✅ Successfully uploaded to: https://huggingface.co/{repo_id}")
        return True
    except Exception as e:
        print(f"❌ Error uploading: {e}")
        return False

if __name__ == "__main__":
    upload_to_huggingface()

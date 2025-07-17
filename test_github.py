#!/usr/bin/env python3
"""Test GitHub connection"""

import os
from dotenv import load_dotenv
import requests

load_dotenv()
token = os.getenv('GITHUB_TOKEN')

# Test GitHub API
headers = {'Authorization': f'token {token}'}
response = requests.get('https://api.github.com/user', headers=headers)

if response.status_code == 200:
    user_data = response.json()
    print(f'✅ GitHub Connected!')
    print(f'👤 Authenticated as: {user_data.get("login", "Unknown")}')
    print(f'📧 Email: {user_data.get("email", "Not public")}')
    print(f'📊 Public repos: {user_data.get("public_repos", 0)}')
else:
    print(f'❌ GitHub connection failed: {response.status_code}')
    print(f'Response: {response.text}')
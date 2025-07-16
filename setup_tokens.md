# How to Safely Store Your Tokens

## IMPORTANT: Revoke Compromised Tokens First!

1. **Revoke GitHub Token**:
   - Visit: https://github.com/settings/tokens
   - Delete the exposed token
   
2. **Revoke Hugging Face Token**:  
   - Visit: https://huggingface.co/settings/tokens
   - Revoke the exposed token

## After Creating New Tokens:

1. Edit the `.env` file in your project root
2. Add your tokens like this:
   ```
   GITHUB_TOKEN=your_new_github_token_here
   HUGGINGFACE_TOKEN=your_new_hf_token_here
   ```

3. The `.env` file is already in `.gitignore` so it won't be committed

## Security Best Practices:
- Never share tokens in messages, code, or commits
- Rotate tokens regularly
- Use tokens with minimal required permissions
- Store tokens only in environment variables or secure vaults
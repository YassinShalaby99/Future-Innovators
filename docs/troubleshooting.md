# Troubleshooting

## Git Push Rejected
Pull remote changes first:
```bash
git pull --rebase origin main
git push
```

## Streamlit Won't Start
- Ensure virtual environment is active
- Install dependencies again:
```bash
pip install -r requirements.txt
```

## Dashboard Empty / Index Error
- Allow runtime to accumulate events
- Validate event loop and window logic

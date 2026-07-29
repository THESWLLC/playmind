# Publish PlayMind as a new GitHub repository

The Cursor cloud agent that bootstrapped this project **did not have GitHub credentials**, so the remote must be created from your account.

## Option A — GitHub CLI (recommended)

```bash
cd /path/to/playmind   # this project root

# one-time
gh auth login

# create + push (private by default; use --public if you want)
gh repo create playmind --private --source=. --remote=origin --push
```

Suggested metadata:
- **Name:** `playmind`
- **Description:** Local self-learning game agent for games you own
- **Visibility:** private while experimenting

## Option B — GitHub website

1. Create an empty repo named `playmind` (no README/license — this project already has them)
2. Then:

```bash
git remote add origin https://github.com/<your-user>/playmind.git
git branch -M main
git push -u origin main
```

If you are on a feature branch:

```bash
git push -u origin HEAD
```

## After it exists

Clone elsewhere:

```bash
git clone https://github.com/<your-user>/playmind.git
cd playmind
python3 playmind_onefile.py
```

## Safety note for the README topics

Keep automation pointed at **your own game**. Automating official WoW (or similar) can violate Blizzard’s EULA and risk account penalties.

# CLAUDE.md

## Python Environment

**Always use the project venv.** Never use system Python.

```bash
# Run any script
.venv/bin/python3 script.py

# Install packages
.venv/bin/pip install <package>
```

The venv is at `.venv/` and uses Python 3.13 (Homebrew).
System Python 3.14 does not have pygame wheels and must not be used.

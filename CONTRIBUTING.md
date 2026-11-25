# Contributing to Sales Forecasting

Thank you for considering contributing to this project! 🎉

## How to Contribute

### 1. Fork & Clone
```bash
git clone https://github.com/Ankitkumar1141/Sales-Forecasting
cd Sales-Forecasting
```

### 2. Set Up the Environment
```bash
pip install -r requirements.txt
cp .env.example secrets.env   # Then fill in your credentials
```

### 3. Create a Feature Branch
```bash
git checkout -b feature/your-feature-name
```

### 4. Make Your Changes
- Keep code modular and consistent with the existing pipeline structure under `src/`.
- Add or update tests in `tests/` for any new functionality.
- Update `config/config.yaml` for any new configurable parameters.
- Run lint before committing: `make lint`

### 5. Run Tests
```bash
make test
```

### 6. Commit & Push
```bash
git commit -m "feat: describe your change"
git push origin feature/your-feature-name
```

### 7. Open a Pull Request
Open a PR against the `main` branch with a clear description of your changes.

## Code Style
- Follow PEP 8. Max line length: 120 characters.
- Use type hints where possible.
- Write docstrings for all public classes and functions.

## Reporting Issues
Please open a GitHub Issue with steps to reproduce, expected behavior, and actual behavior.

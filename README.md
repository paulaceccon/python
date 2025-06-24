# 🐍 Python Learning Playground

This is my personal sandbox for learning, exploring, and experimenting with Python.

Each subfolder is a self-contained project, spike, or study focused on a specific area — whether that's performance tuning, APIs, data manipulation, or just trying out new libraries or patterns.

I use [**Poetry**](https://python-poetry.org/) to manage dependencies and keep environments isolated.

---

## 📦 Environment Setup

### Install Poetry

If you don't have Poetry yet:

```bash
curl -sSL https://install.python-poetry.org | python3 -
```

Make sure it's in your shell path:

```bash
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.zshrc
source ~/.zshrc
```
Check it works:

```bash
poetry --version
```

###  Clone and Install Dependencies

To clone the repo and install dependencies, run:

```bash
git clone https://github.com/your-username/your-project.git
cd your-project
poetry install
```

Activate the environment:

```bash
poetry shell
```

Or run commands inside the environment:

```bash
poetry run python optimized_service_v2.py
```


## 🧪 Pre-commit Configuration
Pre-commit is used for linting, formatting, and enforcing consistency.

To set it up, run:

```bash 
poetry run pre-commit install
```

This will automatically run checks before each commit.

| Folder               | Description                                                                     |
| -------------------- | ------------------------------------------------------------------------------- |
| `perf_optimization/` | Exploring Python performance optimization via async, vector search, and FastAPI |
| `...`                | More to come — this is my Python notebook in repo form                          |

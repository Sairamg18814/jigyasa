# 🤝 Contributing to JIGYASA

First off, thank you for considering contributing to JIGYASA! It's people like you that make JIGYASA such a great tool for the AI community.

## 📋 Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [How Can I Contribute?](#how-can-i-contribute)
- [Development Process](#development-process)
- [Style Guidelines](#style-guidelines)
- [Community](#community)

## 📜 Code of Conduct

This project and everyone participating in it is governed by our Code of Conduct. By participating, you are expected to uphold this code. Please report unacceptable behavior to the maintainers.

## 🚀 Getting Started

1. **Fork the repository** on GitHub
2. **Clone your fork** locally:
   ```bash
   git clone https://github.com/YOUR_USERNAME/jigyasa.git
   cd jigyasa
   ```
3. **Set up the development environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements-dev.txt
   ```
4. **Create a branch** for your feature:
   ```bash
   git checkout -b feature/amazing-feature
   ```

## 💡 How Can I Contribute?

### 🐛 Reporting Bugs

Before creating bug reports, please check existing issues. When you create a bug report, include:

- A clear and descriptive title
- Steps to reproduce the issue
- Expected behavior vs actual behavior
- Screenshots (if applicable)
- Your environment details (OS, Python version, etc.)

### ✨ Suggesting Enhancements

Enhancement suggestions are tracked as GitHub issues. When creating an enhancement suggestion, include:

- A clear and descriptive title
- A detailed description of the proposed enhancement
- Any possible implementations you've considered
- Why this enhancement would be useful

### 🔧 Pull Requests

1. **Follow the style guidelines** (see below)
2. **Write tests** for your changes
3. **Update documentation** as needed
4. **Write clear commit messages**
5. **Include screenshots** for UI changes

#### Pull Request Process

1. Update the README.md with details of changes if needed
2. Ensure all tests pass: `pytest tests/`
3. Update the version numbers following [SemVer](http://semver.org/)
4. The PR will be merged once you have sign-off from maintainers

## 🔄 Development Process

### Branch Naming

- `feature/` - New features
- `fix/` - Bug fixes
- `docs/` - Documentation changes
- `refactor/` - Code refactoring
- `test/` - Test additions or fixes

### Commit Messages

Follow the [Conventional Commits](https://www.conventionalcommits.org/) specification:

```
feat: add new code analysis algorithm
fix: resolve memory leak in continuous learning
docs: update installation instructions
style: format code with black
test: add tests for performance benchmarking
```

## 🎨 Style Guidelines

### Python Style

We use [Black](https://github.com/psf/black) for code formatting:

```bash
black jigyasa/
```

### Code Guidelines

- Use type hints where possible
- Write docstrings for all functions and classes
- Keep functions focused and small
- Use meaningful variable names
- Add comments for complex logic

Example:

```python
def analyze_code(self, code: str) -> Dict[str, Any]:
    """
    Analyze code for potential improvements.
    
    Args:
        code: The source code to analyze
        
    Returns:
        Dictionary containing improvements and metrics
    """
    # Implementation here
```

### Documentation

- Use clear, concise language
- Include code examples
- Keep README and docs up to date
- Add inline comments for complex algorithms

## 🧪 Testing

- Write tests for new features
- Maintain test coverage above 80%
- Use pytest for testing:
  ```bash
  pytest tests/ -v --cov=jigyasa
  ```

## 🌟 Recognition

Contributors will be:
- Added to the Contributors section in README
- Mentioned in release notes
- Given credit in relevant documentation

## 📬 Community

- **Discord**: [Join our server](https://discord.gg/jigyasa)
- **Discussions**: Use GitHub Discussions for questions
- **Twitter**: Follow [@jigyasa_ai](https://twitter.com/jigyasa_ai)

## 🎯 Focus Areas

Current areas where we especially need help:

1. **Multi-language support** (JavaScript, Go, Rust)
2. **Performance optimizations**
3. **Documentation improvements**
4. **Test coverage expansion**
5. **UI/UX for web interface**

---

<div align="center">

**Thank you for contributing to make AI more autonomous! 🚀**

</div>
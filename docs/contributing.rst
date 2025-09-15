Contributing Guide
=================

We welcome contributions to Analytics Toolkit! This guide will help you get started.

Ways to Contribute
------------------

* 🐛 **Report bugs** - Help us identify and fix issues
* 💡 **Suggest features** - Share ideas for improvements
* 📝 **Improve documentation** - Make our docs clearer and more comprehensive
* 🔧 **Submit code** - Fix bugs or implement new features
* 🧪 **Add tests** - Improve our test coverage
* 📚 **Create examples** - Show how to use the toolkit in real scenarios

Getting Started
---------------

Development Setup
~~~~~~~~~~~~~~~~~

1. **Fork the repository** on GitHub
2. **Clone your fork** locally:

   .. code-block:: bash

      git clone https://github.com/your-username/analytics-toolkit.git
      cd analytics-toolkit

3. **Set up development environment**:

   .. code-block:: bash

      # Using our development script
      ./scripts/dev.sh dev-setup

      # Or manually with Poetry
      poetry install --with dev
      poetry run pre-commit install

4. **Create a branch** for your changes:

   .. code-block:: bash

      git checkout -b feature/your-feature-name

Development Workflow
~~~~~~~~~~~~~~~~~~~

1. **Make your changes** in a feature branch
2. **Add tests** for new functionality
3. **Run tests** to ensure everything works:

   .. code-block:: bash

      ./scripts/dev.sh test

4. **Check code quality**:

   .. code-block:: bash

      ./scripts/dev.sh lint

5. **Commit your changes** using conventional commits
6. **Push to your fork** and create a pull request

Code Standards
--------------

Code Style
~~~~~~~~~~

We use several tools to maintain code quality:

* **Black** for code formatting
* **Ruff** for linting and import sorting
* **MyPy** for type checking
* **Pre-commit hooks** for automated checks

Run all checks with:

.. code-block:: bash

   ./scripts/dev.sh lint

Type Hints
~~~~~~~~~~

All public functions and methods should have type hints:

.. code-block:: python

   from typing import Optional, List, Dict, Any
   import pandas as pd

   def process_data(
       data: pd.DataFrame,
       columns: Optional[List[str]] = None,
       **kwargs: Any
   ) -> Dict[str, pd.DataFrame]:
       """Process data with optional column selection."""
       # Implementation here
       pass

Docstrings
~~~~~~~~~~

Use NumPy-style docstrings for all public functions:

.. code-block:: python

   def calculate_metrics(predictions: np.ndarray, targets: np.ndarray) -> Dict[str, float]:
       """Calculate regression metrics.

       Parameters
       ----------
       predictions : np.ndarray
           Model predictions.
       targets : np.ndarray
           True target values.

       Returns
       -------
       Dict[str, float]
           Dictionary containing calculated metrics.

       Examples
       --------
       >>> metrics = calculate_metrics(y_pred, y_true)
       >>> print(metrics['mse'])
       0.123
       """
       pass

Testing Guidelines
------------------

Test Structure
~~~~~~~~~~~~~~

Tests are organized in the ``tests/`` directory:

.. code-block::

   tests/
   ├── test_utils.py
   ├── test_preprocessing.py
   ├── test_models.py
   └── performance/
       └── test_performance.py

Writing Tests
~~~~~~~~~~~~~

Use pytest for testing:

.. code-block:: python

   import pytest
   import pandas as pd
   from analytics_toolkit.utils import describe_data

   def test_describe_data_basic():
       """Test basic functionality of describe_data."""
       data = pd.DataFrame({
           'col1': [1, 2, 3],
           'col2': ['a', 'b', 'c']
       })

       result = describe_data(data)

       assert result['shape'] == (3, 2)
       assert 'col1' in result['columns']
       assert 'col2' in result['columns']

   def test_describe_data_with_missing():
       """Test describe_data with missing values."""
       data = pd.DataFrame({
           'col1': [1, 2, None],
           'col2': ['a', None, 'c']
       })

       result = describe_data(data)

       assert result['missing_values']['col1'] == 1
       assert result['missing_values']['col2'] == 1

Test Coverage
~~~~~~~~~~~~~

We aim for >80% test coverage. Check coverage with:

.. code-block:: bash

   ./scripts/dev.sh test
   # View coverage report
   open htmlcov/index.html

Documentation Guidelines
------------------------

Documentation Structure
~~~~~~~~~~~~~~~~~~~~~~~

Our documentation uses Sphinx with the following structure:

* **API Reference**: Auto-generated from docstrings
* **User Guide**: Tutorials and how-to guides
* **Examples**: Complete code examples
* **Development**: Contributing and development guides

Building Documentation
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   # Build documentation
   ./scripts/dev.sh docs

   # Serve locally
   cd docs/_build/html && python -m http.server 8000

Writing Documentation
~~~~~~~~~~~~~~~~~~~~~

* Use **RST format** for documentation files
* Include **code examples** with expected output
* Add **cross-references** to related sections
* Keep **language simple** and **examples practical**

Pull Request Guidelines
-----------------------

Before Submitting
~~~~~~~~~~~~~~~~~

1. **Ensure all tests pass**: ``./scripts/dev.sh test``
2. **Check code quality**: ``./scripts/dev.sh lint``
3. **Update documentation** if needed
4. **Add changelog entry** for significant changes
5. **Rebase on latest main** branch

PR Description Template
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: markdown

   ## Description
   Brief description of changes and motivation.

   ## Changes
   - [ ] Bug fix (non-breaking change that fixes an issue)
   - [ ] New feature (non-breaking change that adds functionality)
   - [ ] Breaking change (fix or feature that would cause existing functionality to change)
   - [ ] Documentation update

   ## Testing
   - [ ] Added tests for new functionality
   - [ ] All existing tests pass
   - [ ] Manual testing completed

   ## Checklist
   - [ ] Code follows style guidelines
   - [ ] Self-review completed
   - [ ] Documentation updated
   - [ ] Changelog updated (if applicable)

Review Process
~~~~~~~~~~~~~~

1. **Automated checks** must pass (CI/CD pipeline)
2. **Code review** by maintainers
3. **Documentation review** for user-facing changes
4. **Testing verification** for new features
5. **Final approval** and merge

Commit Message Guidelines
-------------------------

We follow the `Conventional Commits <https://www.conventionalcommits.org/>`_ specification:

Format
~~~~~~

.. code-block::

   <type>[optional scope]: <description>

   [optional body]

   [optional footer(s)]

Types
~~~~~

* ``feat``: New feature
* ``fix``: Bug fix
* ``docs``: Documentation changes
* ``style``: Code style changes (formatting, etc.)
* ``refactor``: Code refactoring
* ``test``: Adding or modifying tests
* ``chore``: Maintenance tasks

Examples
~~~~~~~~

.. code-block::

   feat(models): add support for custom loss functions

   - Add CustomLoss base class
   - Update ModelTrainer to accept custom criteria
   - Add tests and documentation

   Closes #123

   fix(utils): handle edge case in describe_data

   - Fix error when DataFrame has zero rows
   - Add test case for empty DataFrames

   refactor(preprocessing): improve DataPreprocessor performance

   - Use vectorized operations instead of loops
   - Reduce memory allocation in fit_transform
   - 20% performance improvement on large datasets

Release Process
---------------

For maintainers, here's the release process:

1. **Update version** in ``pyproject.toml``
2. **Update changelog** with new features and fixes
3. **Create release tag**: ``git tag v1.0.0``
4. **Push tag**: ``git push origin v1.0.0``
5. **GitHub Actions** will automatically:
   - Run full test suite
   - Build and publish to PyPI
   - Create GitHub release
   - Deploy documentation

Issue Guidelines
----------------

Reporting Bugs
~~~~~~~~~~~~~~

When reporting bugs, please include:

* **Python version** and platform
* **Analytics Toolkit version**
* **Minimal code example** that reproduces the issue
* **Expected vs actual behavior**
* **Full error traceback**

Feature Requests
~~~~~~~~~~~~~~~~

For feature requests, please describe:

* **Use case** and motivation
* **Proposed API** or interface
* **Alternative solutions** considered
* **Implementation ideas** (if any)

Community Guidelines
--------------------

We are committed to providing a welcoming and inclusive environment. Please:

* **Be respectful** in all interactions
* **Help others** learn and grow
* **Give constructive feedback**
* **Focus on the code**, not the person
* **Follow our code of conduct**

Recognition
-----------

Contributors are recognized in several ways:

* **Contributors file** in the repository
* **Release notes** mentioning significant contributions
* **Documentation credits** for major documentation work
* **Community highlights** in project updates

Getting Help
------------

If you need help contributing:

* **Check existing issues** and pull requests
* **Join our discussions** on GitHub
* **Ask questions** in issue comments
* **Reach out to maintainers** directly

Thank you for contributing to Analytics Toolkit! 🎉
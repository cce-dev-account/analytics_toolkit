Analytics Toolkit Documentation
================================

.. image:: https://img.shields.io/pypi/v/analytics-toolkit.svg
   :target: https://pypi.org/project/analytics-toolkit/
   :alt: PyPI version

.. image:: https://img.shields.io/pypi/pyversions/analytics-toolkit.svg
   :target: https://pypi.org/project/analytics-toolkit/
   :alt: Python versions

.. image:: https://github.com/your-username/analytics-toolkit/workflows/CI/badge.svg
   :target: https://github.com/your-username/analytics-toolkit/actions
   :alt: CI Status

.. image:: https://codecov.io/gh/your-username/analytics-toolkit/branch/main/graph/badge.svg
   :target: https://codecov.io/gh/your-username/analytics-toolkit
   :alt: Coverage

A comprehensive Python toolkit for data analytics and machine learning with PyTorch support.

Features
--------

* **Data Processing**: Comprehensive data loading, saving, and preprocessing utilities
* **Machine Learning**: PyTorch-based neural networks and training utilities
* **Analytics**: Statistical analysis and data description tools
* **Quality**: Pre-commit hooks, testing, and code formatting

Quick Start
-----------

Installation
~~~~~~~~~~~~

.. code-block:: bash

   pip install analytics-toolkit

Basic Usage
~~~~~~~~~~~

.. code-block:: python

   from analytics_toolkit import utils, preprocessing, models

   # Load and describe data
   data = utils.load_data('data.csv')
   description = utils.describe_data(data)

   # Preprocess data
   preprocessor = preprocessing.DataPreprocessor()
   X_processed, y = preprocessor.fit_transform(data, target_column='target')

   # Create and train a model
   model = models.SimpleNN(
       input_size=X_processed.shape[1],
       hidden_sizes=[64, 32],
       output_size=1
   )
   trainer = models.ModelTrainer(model)

Table of Contents
-----------------

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   installation
   quickstart
   tutorials/index
   examples/index

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api/modules
   api/utils
   api/preprocessing
   api/models

.. toctree::
   :maxdepth: 2
   :caption: Development

   contributing
   development/setup
   development/testing
   development/releases
   changelog

.. toctree::
   :maxdepth: 1
   :caption: About

   license
   authors

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
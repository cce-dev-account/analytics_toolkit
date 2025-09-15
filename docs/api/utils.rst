Utils Module
============

The utils module provides essential data handling utilities for loading, saving, and describing datasets.

.. automodule:: analytics_toolkit.utils
   :members:
   :undoc-members:
   :show-inheritance:

Functions
---------

Data Loading and Saving
~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: analytics_toolkit.utils.load_data

.. autofunction:: analytics_toolkit.utils.save_data

Data Analysis
~~~~~~~~~~~~~

.. autofunction:: analytics_toolkit.utils.describe_data

Examples
--------

Loading Data
~~~~~~~~~~~~

.. code-block:: python

   from analytics_toolkit.utils import load_data

   # Load CSV file
   data = load_data('dataset.csv')

   # Load Parquet file
   data = load_data('dataset.parquet')

   # Load JSON file
   data = load_data('dataset.json')

Saving Data
~~~~~~~~~~~

.. code-block:: python

   from analytics_toolkit.utils import save_data
   import pandas as pd

   # Create sample data
   data = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})

   # Save as CSV
   save_data(data, 'output.csv')

   # Save as Parquet
   save_data(data, 'output.parquet')

Describing Data
~~~~~~~~~~~~~~~

.. code-block:: python

   from analytics_toolkit.utils import describe_data, load_data

   # Load and describe data
   data = load_data('dataset.csv')
   description = describe_data(data)

   print(f"Shape: {description['shape']}")
   print(f"Columns: {description['columns']}")
   print(f"Missing values: {description['missing_values']}")
   print(f"Memory usage: {description['memory_usage']} bytes")
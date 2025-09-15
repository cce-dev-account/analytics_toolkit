Preprocessing Module
===================

The preprocessing module provides comprehensive data preprocessing utilities for machine learning workflows.

.. automodule:: analytics_toolkit.preprocessing
   :members:
   :undoc-members:
   :show-inheritance:

Classes
-------

DataPreprocessor
~~~~~~~~~~~~~~~~

.. autoclass:: analytics_toolkit.preprocessing.DataPreprocessor
   :members:
   :undoc-members:
   :show-inheritance:

Functions
---------

Train-Test Split
~~~~~~~~~~~~~~~~

.. autofunction:: analytics_toolkit.preprocessing.create_train_test_split

Examples
--------

Basic Preprocessing
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from analytics_toolkit.preprocessing import DataPreprocessor
   import pandas as pd

   # Create sample data
   data = pd.DataFrame({
       'numeric_feature': [1.0, 2.0, 3.0, None, 5.0],
       'categorical_feature': ['A', 'B', 'A', 'C', 'B'],
       'target': [0, 1, 0, 1, 1]
   })

   # Initialize preprocessor
   preprocessor = DataPreprocessor()

   # Fit and transform data
   X_processed, y = preprocessor.fit_transform(
       data,
       target_column='target'
   )

   print(f"Processed features shape: {X_processed.shape}")
   print(f"Target shape: {y.shape}")

Advanced Preprocessing
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from analytics_toolkit.preprocessing import DataPreprocessor

   # Initialize with specific scaling method
   preprocessor = DataPreprocessor()

   # Specify column types explicitly
   X_processed, y = preprocessor.fit_transform(
       data,
       target_column='target',
       numerical_columns=['numeric_feature'],
       categorical_columns=['categorical_feature'],
       scaling_method='minmax'
   )

   # Transform new data using fitted preprocessor
   new_data = pd.DataFrame({
       'numeric_feature': [2.5, 4.0],
       'categorical_feature': ['A', 'B']
   })

   X_new = preprocessor.transform(new_data)

Train-Test Splitting
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from analytics_toolkit.preprocessing import create_train_test_split
   import pandas as pd

   # Create sample data
   X = pd.DataFrame({'feature1': range(100), 'feature2': range(100, 200)})
   y = pd.Series(range(100))

   # Split with stratification
   X_train, X_test, y_train, y_test = create_train_test_split(
       X, y,
       test_size=0.2,
       random_state=42,
       stratify=True
   )

   print(f"Training set size: {X_train.shape[0]}")
   print(f"Test set size: {X_test.shape[0]}")

Pipeline Integration
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from analytics_toolkit.preprocessing import DataPreprocessor, create_train_test_split
   from analytics_toolkit.utils import load_data

   # Load data
   data = load_data('dataset.csv')

   # Preprocess
   preprocessor = DataPreprocessor()
   X, y = preprocessor.fit_transform(data, target_column='target')

   # Split
   X_train, X_test, y_train, y_test = create_train_test_split(
       X, y, test_size=0.2, stratify=True
   )

   # Now ready for model training
   print(f"Ready for training with {X_train.shape[0]} samples")
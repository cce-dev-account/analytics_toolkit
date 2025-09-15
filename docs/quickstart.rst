Quick Start Guide
================

This guide will help you get started with Analytics Toolkit in just a few minutes.

First Steps
-----------

1. **Install** Analytics Toolkit (see :doc:`installation`)
2. **Import** the main modules
3. **Load** your data
4. **Analyze** and **model** your data

Basic Example
-------------

Here's a complete example that demonstrates the core workflow:

.. code-block:: python

   import pandas as pd
   from analytics_toolkit import utils, preprocessing, models
   from torch.utils.data import DataLoader

   # 1. Create or load sample data
   data = pd.DataFrame({
       'feature1': [1.0, 2.0, 3.0, 4.0, 5.0],
       'feature2': [2.1, 3.1, 4.1, 5.1, 6.1],
       'category': ['A', 'B', 'A', 'C', 'B'],
       'target': [0.5, 1.5, 1.0, 2.0, 1.8]
   })

   # 2. Describe your data
   description = utils.describe_data(data)
   print(f"Dataset shape: {description['shape']}")
   print(f"Missing values: {description['missing_values']}")

   # 3. Preprocess the data
   preprocessor = preprocessing.DataPreprocessor()
   X, y = preprocessor.fit_transform(data, target_column='target')

   # 4. Split into train/test sets
   X_train, X_test, y_train, y_test = preprocessing.create_train_test_split(
       X, y, test_size=0.2, random_state=42
   )

   # 5. Create datasets and data loaders
   train_dataset = models.TabularDataset(X_train, y_train)
   train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)

   # 6. Create and train a neural network
   model = models.SimpleNN(
       input_size=X_train.shape[1],
       hidden_sizes=[16, 8],
       output_size=1
   )

   trainer = models.ModelTrainer(model)
   history = trainer.train(
       train_loader=train_loader,
       epochs=50,
       learning_rate=0.01
   )

   print(f"Training completed! Final loss: {history['train_loss'][-1]:.4f}")

Core Concepts
-------------

Analytics Toolkit is organized around three main modules:

Utils Module
~~~~~~~~~~~~

The ``utils`` module provides data loading, saving, and analysis functions:

.. code-block:: python

   from analytics_toolkit import utils

   # Load data from various formats
   data = utils.load_data('data.csv')      # CSV
   data = utils.load_data('data.parquet')  # Parquet
   data = utils.load_data('data.json')     # JSON

   # Analyze data
   stats = utils.describe_data(data)

   # Save processed data
   utils.save_data(data, 'processed_data.parquet')

Preprocessing Module
~~~~~~~~~~~~~~~~~~~

The ``preprocessing`` module handles data cleaning and preparation:

.. code-block:: python

   from analytics_toolkit import preprocessing

   # Initialize preprocessor
   preprocessor = preprocessing.DataPreprocessor()

   # Fit and transform data
   X, y = preprocessor.fit_transform(
       data,
       target_column='target',
       scaling_method='standard'  # or 'minmax'
   )

   # Transform new data with same preprocessing
   X_new = preprocessor.transform(new_data)

Models Module
~~~~~~~~~~~~~

The ``models`` module provides PyTorch-based machine learning tools:

.. code-block:: python

   from analytics_toolkit import models
   import torch.nn as nn

   # Create neural network
   model = models.SimpleNN(
       input_size=10,
       hidden_sizes=[64, 32, 16],
       output_size=1,
       dropout_rate=0.2
   )

   # Train the model
   trainer = models.ModelTrainer(model)
   history = trainer.train(
       train_loader=train_loader,
       val_loader=val_loader,
       epochs=100,
       criterion=nn.MSELoss()
   )

Working with Real Data
---------------------

Let's see a more realistic example with a larger dataset:

.. code-block:: python

   import pandas as pd
   import numpy as np
   from analytics_toolkit import utils, preprocessing, models
   from torch.utils.data import DataLoader
   import torch.nn as nn

   # Generate synthetic dataset (replace with your data loading)
   np.random.seed(42)
   n_samples = 1000
   n_features = 10

   data = pd.DataFrame({
       **{f'feature_{i}': np.random.randn(n_samples) for i in range(n_features)},
       'category': np.random.choice(['A', 'B', 'C'], n_samples),
       'target': np.random.randn(n_samples)
   })

   # Add some missing values to simulate real data
   data.loc[data.index[:50], 'feature_0'] = np.nan
   data.loc[data.index[100:120], 'category'] = np.nan

   print(f"Dataset loaded: {data.shape}")
   print(f"Missing values:\n{data.isnull().sum()}")

   # Preprocess the data
   preprocessor = preprocessing.DataPreprocessor()
   X, y = preprocessor.fit_transform(data, target_column='target')

   print(f"After preprocessing: X={X.shape}, y={y.shape}")

   # Split into train/validation/test
   X_train, X_temp, y_train, y_temp = preprocessing.create_train_test_split(
       X, y, test_size=0.4, random_state=42
   )
   X_val, X_test, y_val, y_test = preprocessing.create_train_test_split(
       X_temp, y_temp, test_size=0.5, random_state=42
   )

   # Create data loaders
   train_dataset = models.TabularDataset(X_train, y_train)
   val_dataset = models.TabularDataset(X_val, y_val)
   test_dataset = models.TabularDataset(X_test, y_test)

   train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
   val_loader = DataLoader(val_dataset, batch_size=32)
   test_loader = DataLoader(test_dataset, batch_size=32)

   # Build and train model
   model = models.SimpleNN(
       input_size=X_train.shape[1],
       hidden_sizes=[128, 64, 32],
       output_size=1,
       dropout_rate=0.1
   )

   trainer = models.ModelTrainer(model)

   # Train with validation and early stopping
   history = trainer.train(
       train_loader=train_loader,
       val_loader=val_loader,
       epochs=200,
       learning_rate=0.001,
       criterion=nn.MSELoss(),
       early_stopping_patience=10,
       verbose=True
   )

   # Evaluate on test set
   test_predictions = trainer.predict(test_loader)
   test_loss = nn.MSELoss()(torch.tensor(test_predictions), torch.tensor(y_test.values))
   print(f"Test Loss: {test_loss:.4f}")

Common Patterns
--------------

Data Loading Pipeline
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Load and validate data
   try:
       data = utils.load_data('dataset.csv')
       print(f"✅ Loaded {data.shape[0]} rows, {data.shape[1]} columns")
   except FileNotFoundError:
       print("❌ Data file not found")
       exit(1)

   # Quick data quality check
   description = utils.describe_data(data)
   missing_pct = sum(description['missing_values'].values()) / data.size * 100
   print(f"📊 Missing data: {missing_pct:.1f}%")

Preprocessing Pipeline
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Consistent preprocessing pipeline
   def preprocess_data(data, target_col, test_size=0.2):
       # Split before preprocessing to avoid data leakage
       train_data = data.sample(frac=1-test_size, random_state=42)
       test_data = data.drop(train_data.index)

       # Fit preprocessor on training data only
       preprocessor = preprocessing.DataPreprocessor()
       X_train, y_train = preprocessor.fit_transform(train_data, target_column=target_col)

       # Transform test data with fitted preprocessor
       X_test, y_test = preprocessor.transform(test_data.drop(columns=[target_col])), test_data[target_col]

       return X_train, X_test, y_train, y_test, preprocessor

Model Training Pipeline
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   def train_model(X_train, y_train, X_val, y_val, model_params=None):
       # Default model parameters
       if model_params is None:
           model_params = {
               'hidden_sizes': [64, 32],
               'dropout_rate': 0.2,
               'activation': 'relu'
           }

       # Create model
       model = models.SimpleNN(
           input_size=X_train.shape[1],
           output_size=1,
           **model_params
       )

       # Create data loaders
       train_dataset = models.TabularDataset(X_train, y_train)
       val_dataset = models.TabularDataset(X_val, y_val)

       train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
       val_loader = DataLoader(val_dataset, batch_size=32)

       # Train model
       trainer = models.ModelTrainer(model)
       history = trainer.train(
           train_loader=train_loader,
           val_loader=val_loader,
           epochs=100,
           early_stopping_patience=10
       )

       return model, trainer, history

Next Steps
----------

Now that you've seen the basics, explore these topics:

1. **Advanced Preprocessing**: :doc:`tutorials/preprocessing`
2. **Model Architecture**: :doc:`tutorials/models`
3. **Hyperparameter Tuning**: :doc:`tutorials/tuning`
4. **Production Deployment**: :doc:`tutorials/deployment`
5. **API Reference**: :doc:`api/modules`

Examples Repository
------------------

Check out our `examples repository <https://github.com/your-username/analytics-toolkit-examples>`_ for:

* Complete project templates
* Domain-specific use cases
* Best practices and patterns
* Performance optimization tips

Need Help?
----------

* 📚 Read the full documentation: :doc:`index`
* 💬 Join our community discussions
* 🐛 Report issues on `GitHub <https://github.com/your-username/analytics-toolkit/issues>`_
* 📧 Contact the team: analytics-team@example.com
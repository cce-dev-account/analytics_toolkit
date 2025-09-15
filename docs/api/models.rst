Models Module
=============

The models module provides PyTorch-based machine learning models and training utilities.

.. automodule:: analytics_toolkit.models
   :members:
   :undoc-members:
   :show-inheritance:

Classes
-------

TabularDataset
~~~~~~~~~~~~~~

.. autoclass:: analytics_toolkit.models.TabularDataset
   :members:
   :undoc-members:
   :show-inheritance:

SimpleNN
~~~~~~~~

.. autoclass:: analytics_toolkit.models.SimpleNN
   :members:
   :undoc-members:
   :show-inheritance:

ModelTrainer
~~~~~~~~~~~~

.. autoclass:: analytics_toolkit.models.ModelTrainer
   :members:
   :undoc-members:
   :show-inheritance:

Examples
--------

Creating a Dataset
~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from analytics_toolkit.models import TabularDataset
   import numpy as np

   # Create sample data
   X = np.random.randn(1000, 10)
   y = np.random.randint(0, 2, 1000)

   # Create PyTorch dataset
   dataset = TabularDataset(X, y)

   print(f"Dataset size: {len(dataset)}")
   print(f"First sample: {dataset[0]}")

Building a Neural Network
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from analytics_toolkit.models import SimpleNN
   import torch

   # Define network architecture
   model = SimpleNN(
       input_size=10,
       hidden_sizes=[64, 32, 16],
       output_size=1,
       dropout_rate=0.2,
       activation='relu'
   )

   # Forward pass
   x = torch.randn(32, 10)  # Batch of 32 samples
   output = model(x)
   print(f"Output shape: {output.shape}")

Training a Model
~~~~~~~~~~~~~~~~

.. code-block:: python

   from analytics_toolkit.models import SimpleNN, ModelTrainer, TabularDataset
   from torch.utils.data import DataLoader
   import torch.nn as nn
   import numpy as np

   # Prepare data
   X = np.random.randn(1000, 10)
   y = np.random.randn(1000)

   train_dataset = TabularDataset(X[:800], y[:800])
   val_dataset = TabularDataset(X[800:], y[800:])

   train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
   val_loader = DataLoader(val_dataset, batch_size=32)

   # Create model and trainer
   model = SimpleNN(input_size=10, hidden_sizes=[64, 32], output_size=1)
   trainer = ModelTrainer(model)

   # Train the model
   history = trainer.train(
       train_loader=train_loader,
       val_loader=val_loader,
       epochs=100,
       learning_rate=0.001,
       criterion=nn.MSELoss(),
       early_stopping_patience=10
   )

   # Plot training history
   import matplotlib.pyplot as plt
   plt.plot(history['train_loss'], label='Training Loss')
   plt.plot(history['val_loss'], label='Validation Loss')
   plt.legend()
   plt.show()

Making Predictions
~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from analytics_toolkit.models import ModelTrainer
   from torch.utils.data import DataLoader

   # Assume we have a trained model and trainer
   # Create test dataset
   test_dataset = TabularDataset(X_test)
   test_loader = DataLoader(test_dataset, batch_size=32)

   # Make predictions
   predictions = trainer.predict(test_loader)
   print(f"Predictions shape: {predictions.shape}")

Complete Pipeline Example
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from analytics_toolkit import utils, preprocessing, models
   from torch.utils.data import DataLoader
   import torch.nn as nn

   # 1. Load and preprocess data
   data = utils.load_data('dataset.csv')
   preprocessor = preprocessing.DataPreprocessor()
   X, y = preprocessor.fit_transform(data, target_column='target')

   # 2. Split data
   X_train, X_test, y_train, y_test = preprocessing.create_train_test_split(
       X, y, test_size=0.2, stratify=True
   )

   # 3. Create datasets and loaders
   train_dataset = models.TabularDataset(X_train, y_train)
   val_dataset = models.TabularDataset(X_test, y_test)

   train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
   val_loader = DataLoader(val_dataset, batch_size=32)

   # 4. Create and train model
   model = models.SimpleNN(
       input_size=X_train.shape[1],
       hidden_sizes=[128, 64, 32],
       output_size=1
   )

   trainer = models.ModelTrainer(model)
   history = trainer.train(
       train_loader=train_loader,
       val_loader=val_loader,
       epochs=100,
       criterion=nn.MSELoss()
   )

   # 5. Make predictions
   predictions = trainer.predict(val_loader)
   print(f"Training completed. Final validation loss: {history['val_loss'][-1]:.4f}")

Custom Loss Functions
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import torch
   import torch.nn as nn

   # Custom loss function
   class CustomLoss(nn.Module):
       def __init__(self, alpha=0.5):
           super().__init__()
           self.alpha = alpha
           self.mse = nn.MSELoss()
           self.mae = nn.L1Loss()

       def forward(self, predictions, targets):
           return self.alpha * self.mse(predictions, targets) + (1 - self.alpha) * self.mae(predictions, targets)

   # Use custom loss in training
   custom_criterion = CustomLoss(alpha=0.7)
   history = trainer.train(
       train_loader=train_loader,
       val_loader=val_loader,
       criterion=custom_criterion,
       epochs=50
   )
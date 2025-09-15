Installation
============

System Requirements
-------------------

* Python 3.11 or higher
* pip or Poetry package manager
* Git (for development)

Supported Platforms
~~~~~~~~~~~~~~~~~~~

* Linux (Ubuntu 20.04+, CentOS 8+)
* macOS (10.15+)
* Windows (10+)

Hardware Requirements
~~~~~~~~~~~~~~~~~~~~~

* **Minimum**: 4GB RAM, 2GB free disk space
* **Recommended**: 8GB+ RAM for large datasets
* **GPU**: Optional, PyTorch will automatically use CUDA if available

Installing from PyPI
--------------------

The easiest way to install Analytics Toolkit is using pip:

.. code-block:: bash

   pip install analytics-toolkit

This will install the latest stable version along with all required dependencies.

Installing with Optional Dependencies
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For additional visualization and notebook support:

.. code-block:: bash

   pip install analytics-toolkit[viz,notebooks]

Available optional dependencies:

* ``viz``: Additional visualization libraries (plotly, seaborn)
* ``notebooks``: Jupyter notebook support
* ``dev``: Development dependencies (testing, linting)
* ``all``: All optional dependencies

Installing from Source
----------------------

For the latest development version or to contribute:

.. code-block:: bash

   git clone https://github.com/your-username/analytics-toolkit.git
   cd analytics-toolkit
   pip install -e .

Using Poetry (Recommended for Development)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   git clone https://github.com/your-username/analytics-toolkit.git
   cd analytics-toolkit
   poetry install --with dev

Virtual Environment Setup
-------------------------

It's recommended to use a virtual environment:

Using venv
~~~~~~~~~~

.. code-block:: bash

   python -m venv analytics-env
   source analytics-env/bin/activate  # On Windows: analytics-env\Scripts\activate
   pip install analytics-toolkit

Using conda
~~~~~~~~~~~

.. code-block:: bash

   conda create -n analytics-env python=3.12
   conda activate analytics-env
   pip install analytics-toolkit

Verifying Installation
---------------------

Test your installation:

.. code-block:: python

   import analytics_toolkit
   print(f"Analytics Toolkit version: {analytics_toolkit.__version__}")

   # Test basic functionality
   from analytics_toolkit import utils
   import pandas as pd

   # Create sample data
   data = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
   description = utils.describe_data(data)
   print("✅ Installation successful!")

PyTorch Configuration
--------------------

Analytics Toolkit uses PyTorch for machine learning. The installation will include CPU-only PyTorch by default.

GPU Support
~~~~~~~~~~~

For GPU acceleration, install PyTorch with CUDA support:

.. code-block:: bash

   # Install Analytics Toolkit
   pip install analytics-toolkit

   # Install PyTorch with CUDA (replace cu118 with your CUDA version)
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

Verify GPU support:

.. code-block:: python

   import torch
   print(f"CUDA available: {torch.cuda.is_available()}")
   print(f"CUDA devices: {torch.cuda.device_count()}")

Apple Silicon (M1/M2) Support
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For Apple Silicon Macs with Metal Performance Shaders:

.. code-block:: python

   import torch
   print(f"MPS available: {torch.backends.mps.is_available()}")

Troubleshooting
--------------

Common Issues
~~~~~~~~~~~~~

**Import Error**: ``ModuleNotFoundError: No module named 'analytics_toolkit'``

* Ensure you're in the correct virtual environment
* Verify installation: ``pip list | grep analytics-toolkit``
* Try reinstalling: ``pip install --force-reinstall analytics-toolkit``

**PyTorch Issues**: CUDA/GPU not detected

* Check CUDA installation: ``nvidia-smi``
* Verify PyTorch CUDA version: ``python -c "import torch; print(torch.version.cuda)"``
* Reinstall PyTorch with correct CUDA version

**Memory Issues**: Out of memory during large dataset processing

* Reduce batch size in model training
* Use data chunking for large files
* Monitor memory usage: ``top`` or ``htop``

**Permission Errors**: Access denied during installation

* Use ``--user`` flag: ``pip install --user analytics-toolkit``
* Or install in virtual environment

Getting Help
~~~~~~~~~~~~

If you encounter issues:

1. Check the `FAQ <#faq>`_
2. Search existing `GitHub Issues <https://github.com/your-username/analytics-toolkit/issues>`_
3. Create a new issue with:
   * Python version (``python --version``)
   * Analytics Toolkit version (``pip show analytics-toolkit``)
   * Full error traceback
   * Operating system and version

Upgrading
---------

To upgrade to the latest version:

.. code-block:: bash

   pip install --upgrade analytics-toolkit

Check what's new in the `Changelog <changelog.html>`_.

Uninstalling
-----------

To remove Analytics Toolkit:

.. code-block:: bash

   pip uninstall analytics-toolkit

This will remove the package but keep any data files or notebooks you've created.
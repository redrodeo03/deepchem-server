pyds Library
=============

The pyds library provides a Python client for interacting with DeepChem Server. It offers a convenient, programmatic way to upload data, run featurization tasks, and manage datasets without dealing with raw HTTP requests.

Overview
--------

The pyds library simplifies common workflows by providing:

* **High-level functions** for data upload and featurization
* **Error handling** and retry logic for robust operations
* **Batch processing** capabilities for handling multiple datasets
* **Integration helpers** for popular data science tools

Installation
------------

Install the pyds library using pip:

.. code-block:: bash

   pip install pyds

Or install from source:

.. code-block:: bash

   cd pyds
   pip install -e .

Configuration
-------------

Configure the library to connect to your DeepChem Server using environment variables, configuration files, or programmatically. For detailed configuration options, refer to the module documentation.

Interactive API Testing
-----------------------

For the most up-to-date examples and interactive testing of the underlying API that the pyds library uses, visit http://localhost:8000/docs when your DeepChem Server is running.

The Swagger UI provides:

* **Live endpoint testing**: Test API calls interactively
* **Request/response examples**: See real data formats
* **Parameter documentation**: Understand all available options
* **Schema definitions**: Review data structures

Library Documentation
---------------------

For detailed documentation of the pyds library functions and classes, refer to the auto-generated documentation from the source code docstrings.

Getting Started
---------------

1. **Install the library**: Follow the installation instructions above
2. **Configure connection**: Set up connection to your DeepChem Server
3. **Explore the API**: Visit http://localhost:8000/docs to understand available endpoints
4. **Check examples**: Review the examples section for common workflows
5. **Read the source**: Examine the library source code for detailed implementation

For comprehensive examples, testing, and up-to-date API information, always refer to:

* **Interactive docs**: http://localhost:8000/docs
* **Source code**: Check the pyds directory for implementation details
* **Module documentation**: Auto-generated docs from docstrings 
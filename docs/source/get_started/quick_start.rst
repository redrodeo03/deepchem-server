Quick Start
===========

This guide will walk you through your first steps with DeepChem Server, from uploading a dataset to performing molecular featurization.

Before You Begin
----------------

Make sure you have:

1. DeepChem Server running (see :doc:`installation`)
2. A sample dataset (CSV file with SMILES strings)

Basic Workflow
--------------

The typical workflow with DeepChem Server involves:

1. **Upload Data**: Submit your molecular dataset
2. **Featurize**: Transform molecules into feature vectors
3. **Retrieve Results**: Download the featurized dataset

Interactive API Documentation
-----------------------------

Once your DeepChem Server is running, the best way to explore and test the API is through the interactive documentation:

**Swagger UI**: http://localhost:8000/docs

This provides:

* Complete endpoint documentation
* Interactive request testing
* Response examples
* Parameter descriptions
* Schema definitions

Available Endpoints
-------------------

The main endpoints you'll work with are:

* **POST /data/uploaddata**: Upload datasets
* **POST /primitive/featurize**: Apply molecular featurization
* **GET /healthcheck**: Check server status

Available Featurizers
---------------------

DeepChem Server supports the following molecular featurizers:

.. list-table::
   :header-rows: 1
   :widths: 20 30 50

   * - Featurizer
     - DeepChem Class
     - Description
   * - ``ecfp``
     - ``CircularFingerprint``
     - Extended Connectivity Fingerprints (Morgan/ECFP)
   * - ``graphconv``
     - ``ConvMolFeaturizer``
     - Graph convolution molecular featurizer
   * - ``weave``
     - ``WeaveFeaturizer``
     - Weave molecular featurizer for graph networks
   * - ``molgraphconv``
     - ``MolGraphConvFeaturizer``
     - Molecular graph convolution featurizer

For detailed information about these featurizers, including parameters and usage examples, refer to the `DeepChem Featurizers documentation <https://deepchem.readthedocs.io/en/latest/api_reference/featurizers.html>`_.

For parameter information and interactive testing, visit http://localhost:8000/docs

Next Steps
----------

Now that you understand the basic workflow:

1. **Visit the Interactive Docs**: Go to http://localhost:8000/docs to test endpoints
2. **Try Different Featurizers**: Experiment with various molecular representations
3. **Use the Python Client**: Explore the pyds library for programmatic access
4. **Consult DeepChem Documentation**: For detailed featurizer information, see the `DeepChem Featurizers documentation <https://deepchem.readthedocs.io/en/latest/api_reference/featurizers.html>`_

Troubleshooting Quick Start
---------------------------

**Server Not Responding**
   Check if the server is running: ``curl http://localhost:8000/healthcheck``

**Need More Information**
   Visit http://localhost:8000/docs for comprehensive API documentation and interactive testing. 
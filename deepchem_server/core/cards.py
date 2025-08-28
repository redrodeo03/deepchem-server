import ast
import datetime
import json
from typing import Dict, Optional

from deepchem_server.core import model_mappings

class Card:
    """Base class for cards.

    Provides common functionality for data and model cards including
    serialization and timestamp tracking.
    """

    def __init__(self) -> None:
        """Initialize a Card with current timestamp."""
        self.last_updated_time = datetime.datetime.now().strftime('%d-%B-%Y %H:%M:%S')

    def __bytes__(self) -> bytes:
        """Convert card to bytes representation.

        Returns
        -------
        bytes
            The card as bytes using UTF-8 encoding.
        """
        return bytes(self.to_json(), encoding='utf8')

    def to_json(self) -> str:
        """Convert card to JSON string representation.

        Returns
        -------
        str
            JSON string representation of the card.
        """
        return json.dumps(self, default=lambda o: o.__dict__)

    def update_card(self, key: str, value) -> None:
        """Update a card attribute.

        Parameters
        ----------
        key : str
            The attribute name to update.
        value : Any
            The new value for the attribute.

        Returns
        -------
        None
        """
        setattr(self, key, value)

class DataCard(Card):
    """Class for storing data card attributes.

    Parameters
    ----------
    address : str
        Address of the reference object in the datastore.
    file_type : str
        The file extension - ex. csv filetype, .json file type etc.
    data_type : str
        The type of object stored at the location pointed by filename -
        ex: pd.DataFrame, dask.dataframe.DataFrame.
    shape : tuple, optional
        Shape of the data object.
    description : str, optional
        A description about the datastore.
    featurizer : str, optional
        The featurizer used in the dataset.
    intended_use : str, optional
        Notes on dataset - the intended use of the dataset.
    caveats : str, optional
        Notes on dataset - the caveats in using the dataset.
    feat_kwargs : dict, optional
        Keyword arguments for featurizer (used when featurizer is not None).
    **kwargs
        Additional attributes to set on the card.

    Notes
    -----
    Difference between data_type and file_type:
    An example can illustrate this better. A csv file (file_type) can either be
    a pandas.DataFrame or dask.dataframe.DataFrame or just a csv file. The file_type
    holds the file extension ('csv') while data_type refers to the data
    object (pandas.DataFrame, dask.dataframe.DataFrame, etc).
    """
    SUPPORTED_DATA_TYPES = [
        'pandas.DataFrame',
        'dc.data.NumpyDataset',
        'dc.data.DiskDataset',
        'json',
        'text/plain',
        'png',
        'binary',
    ]
    SUPPORTED_FILE_TYPES = [
        'csv', 'dir', 'json', 'pdb', 'fasta', 'fastq', 'png', 'sdf', 'dcd', 'txt', 'xml', 'py', 'pdbqt', 'zip', 'smi',
        'smiles', 'bz2', 'cxsmiles', 'onnx', 'hdf5', 'log'
    ]

    def __init__(self,
                 address: str,
                 file_type: str,
                 data_type: str,
                 shape=None,
                 description: Optional[str] = None,
                 featurizer: Optional[str] = None,
                 intended_use: Optional[str] = None,
                 caveats: Optional[str] = None,
                 feat_kwargs: Optional[Dict] = None,
                 **kwargs) -> None:
        """Initialize a DataCard."""
        super().__init__()
        data_type = self.validate_datatype(data_type)
        if not isinstance(address, str):
            raise TypeError("address must be a string")
        if not isinstance(file_type, str):
            raise TypeError("file_type must by a string")
        if not isinstance(data_type, str):
            raise TypeError("data_type must be a string")
        assert file_type in self.SUPPORTED_FILE_TYPES, 'Filetype {} is not supported. Supported data type are {}'.format(
            file_type, self.SUPPORTED_FILE_TYPES)
        self.shape = shape
        self.address = address
        self.file_type = file_type
        self.data_type = data_type
        self.description = description
        self.featurizer = featurizer
        self.intended_use = intended_use
        self.caveats = caveats
        self.feat_kwargs = feat_kwargs
        # FIXME we should not depend on kwargs as internal parameters. These are features
        # for users to store additional details.
        for key, value in kwargs.items():
            setattr(self, key, value)

    def validate_datatype(self, data_type: str) -> str:
        """Validate and normalize data type name.

        Parameters
        ----------
        data_type : str
            The data type to validate.

        Returns
        -------
        str
            The validated and normalized data type.

        Raises
        ------
        AssertionError
            If the data type is not supported.
        """
        # expand data type name
        if data_type == 'DataFrame':
            data_type = 'pandas.DataFrame'
        elif data_type == 'DiskDataset':
            data_type = 'dc.data.DiskDataset'
        elif data_type == 'NumpyDataset':
            data_type = 'dc.data.NumpyDataset'
        assert data_type in self.SUPPORTED_DATA_TYPES, 'Datatype {} is not supported. Supported file type are {}'.format(
            data_type, self.SUPPORTED_DATA_TYPES)
        return data_type

    @classmethod
    def from_json(cls, json_data: str) -> "DataCard":
        """Create DataCard from JSON string.

        Parameters
        ----------
        json_data : str
            JSON string representation of the DataCard.

        Returns
        -------
        DataCard
            DataCard instance created from the JSON data.
        """
        args = json.loads(json_data)
        return cls(**args)
        # Note: The above method may fail if `DataCard` contains nested objects.
        # return json.loads(json_data, object_hook=lambda d: DataCard(**d))

    @classmethod
    def from_bytes(cls, card_bytes: bytes) -> "DataCard":
        """Create DataCard from bytes.

        Parameters
        ----------
        card_bytes : bytes
            Bytes representation of the DataCard.

        Returns
        -------
        DataCard
            DataCard instance created from the bytes data.
        """
        return DataCard.from_json(card_bytes.decode('utf8'))

    def get_n_samples(self) -> int:
        """Get the number of samples in the dataset.

        Returns
        -------
        int
            Number of samples in the dataset.

        Raises
        ------
        ValueError
            If the dataset does not have shape information.
        """
        if self.shape is None:
            raise ValueError("the dataset does not have shape")
        if self.data_type == 'pandas.DataFrame':
            return self.shape[0]  # shape of dataframe is (n_rows x n_cols)
        elif self.data_type in ['dc.data.DiskDataset', 'dc.data.NumpyDataset']:
            # A deepchem dataset has shape (X_shape, y_shape, w_shape, ids_shape)
            x_shape = self.shape[0]
            return x_shape[0]
        else:
            return self.shape

    def to_json(self) -> str:
        """Convert DataCard to JSON string.

        Returns
        -------
        str
            JSON string representation of the DataCard.
        """
        json_str = json.dumps(self, default=lambda o: o.__dict__)
        return json_str.replace('_shape', 'shape')

    @property
    def shape(self):
        """Get the shape of the data.

        Returns
        -------
        tuple
            Shape of the data as a tuple.
        """
        # FIXME This might pose security risk if user
        # arbitrarily sets card shape
        return ast.literal_eval(self._shape)

    @shape.setter
    def shape(self, value) -> None:
        """Set the shape of the data.

        Parameters
        ----------
        value : tuple or None
            Shape of the data to set.
        """
        self._shape = str(value)

class ModelCard(Card):
    """Class for storing model card attributes.

    Parameters
    ----------
    address : str
        The address of model in the datastore.
    model_type : str
        The type of model. Ex: dc.models.RandomForest.
    train_dataset_address : str
        Training dataset used to train the model.
    description : str, optional
        A description about the model.
    featurizer : str, optional
        The featurizer used in the dataset.
    intended_use : str, optional
        Notes on dataset - the intended use of the dataset.
    caveats : str, optional
        Notes on dataset - the caveats in using the dataset.
    init_kwargs : dict, optional
        Initialization kwargs for the model ex: n_layers.
    train_kwargs : dict, optional
        Training kwargs for the model ex: n_epochs.
    **kwargs
        Additional attributes to set on the model card.
    """
    SUPPORTED_MODEL_TYPES = list(model_mappings.model_address_map.keys())

    def __init__(self,
                 address: str,
                 model_type: str,
                 train_dataset_address: str,
                 description: Optional[str] = None,
                 featurizer: Optional[str] = None,
                 intended_use: Optional[str] = None,
                 caveats: Optional[str] = None,
                 init_kwargs: Optional[Dict] = {},
                 train_kwargs: Optional[Dict] = {},
                 **kwargs) -> None:
        """Initialize a ModelCard."""
        super().__init__()
        if not isinstance(address, str):
            raise TypeError("address must be a string")
        if not isinstance(model_type, str):
            raise TypeError("model_type must by a string")
        if not isinstance(train_dataset_address, str):
            raise TypeError("train_dataset_address must be a string")
        assert model_type in self.SUPPORTED_MODEL_TYPES, 'Model type {} is not supported. Supported model types are {}'.format(
            model_type, ' '.join(self.SUPPORTED_MODEL_TYPES))
        self.address = address
        self.model_type = model_type
        self.train_dataset_address = train_dataset_address
        # Note: We don't have datatype here because we will be only storing
        # models of type dc.model.Models
        self.description = description
        self.featurizer = featurizer
        self.intended_use = intended_use
        self.caveats = caveats
        self.init_kwargs = init_kwargs
        self.train_kwargs = train_kwargs
        self.pretrained_model_address: Optional[str] = None
        for key, value in kwargs.items():
            setattr(self, key, value)

    @classmethod
    def from_json(cls, json_data: str) -> "ModelCard":
        """Create ModelCard from JSON string.

        Parameters
        ----------
        json_data : str
            JSON string representation of the ModelCard.

        Returns
        -------
        ModelCard
            ModelCard instance created from the JSON data.
        """
        data = json.loads(json_data)
        return cls(**data)

    @classmethod
    def from_bytes(cls, card_bytes: bytes) -> "ModelCard":
        """Create ModelCard from bytes.

        Parameters
        ----------
        card_bytes : bytes
            Bytes representation of the ModelCard.

        Returns
        -------
        ModelCard
            ModelCard instance created from the bytes data.
        """
        return ModelCard.from_json(card_bytes.decode('utf8'))

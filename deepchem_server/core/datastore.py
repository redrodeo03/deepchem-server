# mypy: ignore-errors
# mypy errors ignored because ModelCard yet to added
from __future__ import annotations

import logging
import os
from pathlib import Path
import shutil
from typing import Any, IO, List, Optional, Tuple, Union

import deepchem as dc
import pandas as pd
from PIL import Image
from PIL.PngImagePlugin import PngImageFile

from deepchem_server.core import model_mappings
from deepchem_server.core.address import DeepchemAddress
from deepchem_server.core.cards import Card, DataCard, ModelCard  # yapf: disable


try:
    import mdtraj as md
except ModuleNotFoundError:
    pass

logger = logging.getLogger(__name__)

# List of kinds supported by deepchem server, used to determine whether a file is a card or not and to determine the kind of the object
KIND_LIST = [{
    'name': 'data',
    'extension': '.cdc',
}, {
    'name': 'model',
    'extension': '.cmc',
}]

# Number of rows to get when fetching a sample instead of full data (works only for csv)
DEFAULT_SAMPLE_ROWS = 100  # for disk datastore


def _get_csv_or_dataframe_shape(*,
                                filename: Optional[str] = None,
                                dataframe: Optional[pd.DataFrame] = None) -> Tuple[int, int]:
    """Get the shape of a CSV file or pandas DataFrame.

    Parameters
    ----------
    filename : str, optional
        Path to the CSV file.
    dataframe : pd.DataFrame, optional
        The pandas DataFrame to get shape from.

    Returns
    -------
    tuple of (int, int)
        The shape as (number of rows, number of columns).

    Raises
    ------
    ValueError
        If neither filename nor dataframe is provided.
    """
    if filename is None and dataframe is None:
        raise ValueError("Either one of filepath or dataframe should be set")
    if filename is not None:
        numrows = sum(1 for line in open(filename)) - 1
        with open(filename, 'r') as f:
            line = f.readline()
        # a rough heuristic for number of columns in a csv file
        numcols = line.count(',') + 1
    elif dataframe is not None:
        numrows = dataframe.shape[0]
        numcols = dataframe.shape[1]
    return (numrows, numcols)


class DataStore:
    """Python API wrapper for deepchem server Backend data.

    Each user of deepchem server has access to a personal backend
    datastore. The datastore is used to hold uploaded datasets
    and trained models. Users may refer to objects by their
    deepchem server datastore address and can use the deepchem server API to
    download them.

    This abstract superclass provides a common datastore API
    that will be used to govern concrete Datastore
    implementations.
    """

    def upload_data(self, datastore_filename: Any, filename: Any, card: Union[ModelCard, DataCard]) -> Optional[str]:
        """Upload data to the datastore in question.

        Parameters
        ----------
        datastore_filename : Any
            The name of this dataset within your deepchem server datastore.
        filename : Any
            Should be the location of a file on disk that is to be uploaded.
        card : ModelCard or DataCard
            The card containing metadata for the uploaded data.

        Returns
        -------
        str or None
            If request failed, returns None. Else returns the deepchem server
            dataset address for the dataset in question.
        """
        raise NotImplementedError

    def get(self, deepchem_address: str, kind: Optional[str], fetch_sample: bool):
        """Fetch something from datastore at address.

        Parameters
        ----------
        deepchem_address : str
            Should be the location of a file on deepchem server datastore.
        kind : str, optional
            'data' or 'model' - used in cases which contain data in a directory
            and we need to find the contents of the directory as data or model.
        fetch_sample : bool
            Whether to get sample or full data.

        Returns
        -------
        Any
            The requested data or model object.
        """
        raise NotImplementedError

    def delete_object(self, deepchem_address: str):
        """Delete an object pointed by the address from the datastore.

        Parameters
        ----------
        deepchem_address : str
            Location of object in the datastore.

        Returns
        -------
        Any
            Result of the deletion operation.
        """
        raise NotImplementedError

    # TODO Add list_model utility
    def list_data(self):
        """List data uploaded to deepchem server datastore.

        This method lists data that is present in deepchem server datastore
        for the present user.

        Returns
        -------
        Any
            Representation of available data in the datastore.
        """
        raise NotImplementedError


class DiskDataStore(DataStore):
    """A concrete datastore that stores objects on the local disk."""

    def __init__(
        self,
        profile_name: str,
        project_name: str,
        basedir: Optional[str] = None,
        sample_rows: int = DEFAULT_SAMPLE_ROWS,
    ) -> None:
        """Initialize a disk datastore within the given directory.

        Parameters
        ----------
        profile_name : str
            Name of the profile.
        project_name : str
            Name of the project.
        basedir : str, optional
            Location on disk to hold data store. If none, create temporary folder.
        sample_rows : int, optional
            Number of rows to get when fetching a sample instead of full data
            (works only for csv), by default {DEFAULT_SAMPLE_ROWS}.
        """
        if basedir:
            self.storage_loc = os.path.join(basedir, profile_name, project_name)
        else:
            self.storage_loc = os.path.join(profile_name, project_name)
        self.address_prefix = profile_name + '/' + project_name + '/'
        if not os.path.exists(self.storage_loc):
            os.makedirs(self.storage_loc)
        objects = self._get_datastore_objects(self.storage_loc)
        self._objects = objects
        self.sample_rows = sample_rows

    def _get_datastore_objects(self, directory: str) -> List[str]:
        """Walk directory structure and collect all objects.

        It walks the root directory structure and collects all the objects
        in it including files and subfolder names.

        Parameters
        ----------
        directory : str
            The directory to walk through.

        Returns
        -------
        list of str
            List of relative paths to all files and directories.
        """
        # TODO We should also list objects in the common namespace which
        # can be used by the user
        entries = []
        for root, dirs, files in os.walk(directory):
            for dir in dirs:
                # relpath removes main directory info from path to avoid repetition in list data
                relative_path = os.path.relpath(os.path.join(root, dir), directory)
                entries.append(relative_path + "/")
            for file in files:
                # relpath removes main directory info from path to avoid repetition in list data
                relative_path = os.path.relpath(os.path.join(root, file), directory)
                entries.append(relative_path)
        return entries

    def upload_data_from_memory(
        self,
        data: Any,
        datastore_filename: str,
        card: Union[DataCard, ModelCard, None],
        kind: str = "data",
    ) -> str:
        """Upload in memory data to filestore.

        Parameters
        ----------
        data : Any
            Dataset to upload (ex: dataframe, image dataset etc).
        datastore_filename : str
            The name of this dataset within your deepchem server datastore.
        card : DataCard, ModelCard, or None
            Description of dataset for the dataset card.
        kind : str, optional
            Type of data being uploaded, by default 'data'.

        Returns
        -------
        str or None
            If request failed, returns None. Else returns the deepchem server
            dataset address for the dataset in question.

        Raises
        ------
        ValueError
            If unsupported data type is provided.
        FileExistsError
            If the file name already exists in the datastore.
        """
        dataset_address = DeepchemAddress(self.address_prefix + datastore_filename).address
        dest_loc = os.path.join(self.storage_loc, datastore_filename)

        dir_path = os.path.dirname(dest_loc)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

        if isinstance(data, Card):
            with open(dest_loc, 'wb') as fp:
                fp.write(bytes(data))
            return repr(dataset_address)
        if card is not None:
            card.address = dataset_address

        if isinstance(card, DataCard) and isinstance(data, pd.DataFrame):
            card.shape = _get_csv_or_dataframe_shape(dataframe=data)
            data.to_csv(path_or_buf=dest_loc, index=False)
        elif isinstance(card, DataCard) and isinstance(data, dc.data.NumpyDataset):
            # This writes to disk
            card.shape = data.get_shape()
            dc.data.DiskDataset.from_numpy(data.X, data.y, data.w, data.ids, data_dir=dest_loc)
        elif isinstance(data, dc.models.Model):
            shutil.copytree(data.model_dir, dest_loc)
        elif isinstance(card, DataCard) and isinstance(data, dc.data.DiskDataset):
            card.shape = data.get_shape()
            try:
                shutil.copytree(data.data_dir, dest_loc)
            except FileExistsError:
                raise FileExistsError(f"File name '{datastore_filename}' already exists!")
        elif isinstance(data, str):
            with open(dest_loc, 'w') as fp:
                fp.write(data)
        elif isinstance(data, bytes):
            with open(dest_loc, 'wb') as fp:
                fp.write(data)
        elif isinstance(data, PngImageFile):
            data.save(dest_loc)
        else:
            raise ValueError("Only dataframes, deepchem datasets, and models are supported for now")

        if kind == 'data':
            card_path = dest_loc + '.cdc'
        elif kind == 'model':
            card_path = dest_loc + '.cmc'
        else:
            raise ValueError(f"Unsupported kind '{kind}' provided. "
                             f"Supported kinds are {[k['name'] for k in KIND_LIST]}")
        if card is not None:
            with open(card_path, 'wb') as fp:
                fp.write(bytes(card))

        return dataset_address

    def upload_data(self,
                    datastore_filename: str,
                    filename,
                    card: Union[ModelCard, DataCard],
                    kind: Optional[str] = 'data') -> str:
        """Upload data to DiskDataStore

        Parameters
        ----------
        datastore_filename: str
          The name of this dataset within your deepchem server datastore.
        filename: str
          Should be the location of a file or directory on disk that is to be uploaded.
        card: ModelCard or DataCard
            The card containing metadata for the uploaded data.
        kind: Optional[str]
            Type of data being uploaded, by default 'data'.

        Returns
        -------
        dataset_address: Optional[str]
           If request failed, returns None. Else returns the deepchem server
           dataset address for the dataset in question.
        """
        dataset_address = DeepchemAddress(self.address_prefix + datastore_filename).address
        card.address = dataset_address
        dest_loc = os.path.join(self.storage_loc, datastore_filename)
        if isinstance(card, DataCard) and datastore_filename.endswith('.csv'):
            card.shape = _get_csv_or_dataframe_shape(filename=filename)

        dir_path = os.path.dirname(dest_loc)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

        if os.path.isdir(filename):
            try:
                shutil.copytree(filename, dest_loc)
            except FileExistsError:
                raise FileExistsError(f"File name '{datastore_filename}' already exists!")
        elif os.path.isfile(filename):
            shutil.copyfile(filename, dest_loc)
        elif isinstance(filename, bytes):
            with open(dest_loc, 'wb') as f:
                f.write(filename)
        elif isinstance(filename, str):
            with open(dest_loc, 'w') as f:
                f.write(filename)
        else:
            raise ValueError("filename must be either a file or directory.")

        # Write data card
        card_path = dest_loc + '.cdc'
        with open(card_path, 'wb') as fp:
            fp.write(bytes(card))
        return dataset_address

    def add_dir(self, dir_name: str):
        """
        Adds a directory to the DiskDataStore

        Parameters
        ----------
        dir_name: str
          Name of the directory to be added
        ------

        """
        key = os.path.join(self.storage_loc, dir_name)
        if os.path.exists(key):
            raise ValueError(f"Directory '{dir_name}' already exists.")
        else:
            os.makedirs(key)

    def list_data(self):
        """Lists data uploaded to deepchem server datastore.

        This method lists data that is present in deepchem server datastore
        for the present user.
        """
        return repr(self)

    def upload_model(self, modelname: str, model, card: ModelCard):  # noqa
        """Upload model data to DiskDataStore

        Parameters
        ----------
        modelname: str
            The name of the model in datastore.
        model: dc.model.Model
            Model which is to be uploaded to datastore
        card: str
            Description of model for the model card

        Returns
        -------
        model_address: Optional[str]
           If request failed, returns None. Else returns the deepchem server
           model address for the uploaded model.
        """
        model_address = DeepchemAddress(self.address_prefix + modelname).address
        card.address = model_address
        dest_loc = os.path.join(self.storage_loc, modelname)

        if os.path.exists(dest_loc):
            raise FileExistsError(f"Model '{modelname}' already exists!")

        dir_path = os.path.dirname(dest_loc)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

        assert isinstance(model, dc.models.Model), 'Model must be a deepchem model'
        shutil.copytree(model.model_dir, dest_loc)

        # Write Model card
        card_path = dest_loc + '.cmc'
        with open(card_path, 'wb') as fp:
            fp.write(bytes(card))
        return model_address

    def get_dir(self, address: str) -> str:
        """
        Returns the directory of the object

        Parameters
        ----------
        address: str
          DeepchemAddress of the object

        Returns
        -------
        _dir: str
          Directory of the object
        """
        key = DeepchemAddress.get_key(address)
        _dir = os.path.join(self.storage_loc, key)
        if os.path.isdir(_dir):
            return _dir
        else:
            raise FileNotFoundError(f"Directory {_dir} not found")

    def upload_model_from_memory(self, model_name: str, model_files: List[IO], model_filenames: List[str],
                                 card: ModelCard) -> Union[str, None]:
        """Upload model data to DiskDataStore

        Parameters
        ----------
        model_name: str
            The name of the model in datastore.
        model_files: List[IO]
            List of file-like objects containing model files
        model_filenames: List[str]
            List of filenames for the model files
        card: str
            Description of model for the model card

        Returns
        -------
        model_address: Optional[str]
            If request failed, returns None. Else returns the deepchem server
            model address for the uploaded model.
        """
        model_address = DeepchemAddress(self.address_prefix + model_name).address
        card.address = model_address
        dest_loc = os.path.join(self.storage_loc, model_name)
        dir_path = os.path.dirname(dest_loc)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        if not os.path.exists(dest_loc):
            os.makedirs(dest_loc)
        for model_file, model_filename in zip(model_files, model_filenames):
            with open(os.path.join(dest_loc, model_filename), 'wb') as f:
                f.write(model_file.read())

        # Write Model card
        card_path = dest_loc + '.cmc'
        with open(card_path, 'wb') as fp:
            fp.write(bytes(card))
        return model_address

    def get_card(self, address: str, kind: Optional[str] = "data") -> Optional[Union[DataCard, ModelCard]]:
        """Fetch card from disk data store at address.

        Parameters
        ----------
        address : str
            DeepchemAddress of the data object to retrieve.
        kind : str, optional
            'data' or 'model' - used in cases which contain data in a directory
            and we need to find the contents of the directory as data or model,
            by default 'data'.

        Returns
        -------
        DataCard, ModelCard, or None
            The card object if found, None otherwise.
        """
        if kind == 'data':
            address = address + '.cdc'
        elif kind == 'model':
            address = address + '.cmc'
        address_key = DeepchemAddress.get_key(address)
        path = os.path.join(self.storage_loc, address_key)

        if path.endswith('.cdc'):
            with open(path, 'r') as f:
                card_data = f.readlines()
            card = DataCard.from_json(card_data[0])
            return card
        if path.endswith('.cmc'):
            with open(path, 'r') as f:
                card_data = f.readlines()
            card = ModelCard.from_json(card_data[0])
            return card
        return None

    def get_data(self, address, fetch_sample: bool = False):
        """Fetch data from disk data store at address

        Parameters
        ----------
        address
            DeepchemAddress of the data object to retrieve
        fetch_sample: bool
            Whether to get sample or full data (currently works only for csv files)
        """
        # TODO Check whether key exists
        address_key = DeepchemAddress.get_key(address)
        path = os.path.join(self.storage_loc, address_key)
        card = self.get_card(address, kind='data')
        if card is not None and isinstance(card, DataCard):
            if card.file_type == 'csv':
                if fetch_sample:
                    df = pd.read_csv(path, nrows=self.sample_rows)
                else:
                    df = pd.read_csv(path)
                return df
            elif card.file_type == 'pdb':
                out = md.load_pdb(path)
                return out
            elif card.file_type == 'pdbqt':
                with open(path, 'r') as f:
                    data = f.readlines()
                    return data
            elif card.file_type == 'fasta':
                with open(path, 'r') as f:
                    data = f.readlines()
                return data
            elif card.file_type == 'fastq':
                with open(path, 'r') as f:
                    data = f.readlines()
            elif card.file_type == 'json':
                # Return parsed JSON if possible; fall back to raw string
                with open(path, 'r') as f:
                    text = f.read()
                try:
                    import json as _json
                    return _json.loads(text)
                except Exception:
                    return text
            elif card.file_type == 'txt':
                with open(path, 'r') as f:
                    data = f.readlines()
                    return data
            elif card.file_type == 'dcd':
                import MDAnalysis as mda
                dcd = mda.coordinates.LAMMPS.DCDReader(path)
                return dcd
            elif card.file_type == 'xml':
                with open(path, 'r') as f:
                    data = f.readlines()
                return data
            elif card.data_type == 'dc.data.DiskDataset':
                dataset = dc.data.DiskDataset(data_dir=path)
                return dataset
            elif card.data_type == 'sdf':
                with open(path, 'r') as f:
                    data = f.readlines()
                return data
            elif card.data_type == 'png':
                im = Image.open(path)
                return im
        return None

    def get_model(self, address):
        """Fetch model from disk data store at address

        Parameters
        ----------
        address: DeepchemAddress of the data object to retrieve
        """
        address_key = DeepchemAddress.get_key(address)
        path = os.path.join(self.storage_loc, address_key)

        model_card = self.get_card(address, kind='model')

        model = model_mappings.model_address_map[model_card.model_type](  # noqa
            model_dir=path, **model_card.init_kwargs)
        try:
            model.restore()
        except AttributeError:
            model.reload()
        return model

    def get(self, address, kind: Optional[str] = 'data', fetch_sample: bool = False):
        """Fetch something from disk datastore at address.

        Parameters
        ----------
        address
            DeepchemAddress of the data object to retrieve
        kind: Optional[str]
            'data' or 'model' - used in cases which contain data in a directory
            and we need to find the contents of the directory as data or model
        fetch_sample: bool
            Whether to get sample or full data (currently works only for csv files)
        """
        # TODO Check whether key exists
        if address.endswith('.cdc'):
            return self.get_card(address[:-4], kind='data')  # [:-4] removes ".cdc" from address string
        elif address.endswith('.cmc'):
            return self.get_card(address[:-4], kind='model')  # [:-4] removes ".cmc" from address string
        if kind == 'data':
            dataset = self.get_data(address, fetch_sample)
            return dataset
        elif kind == 'model':
            model = self.get_model(address)  # noqa
            return model
        return None

    def get_file_size(self, address: str) -> int:
        """Return size of the object.

        Parameters
        ----------
        address : str
            DeepchemAddress of the object.

        Returns
        -------
        int
            Size of the object in bytes.
        """
        address_key = DeepchemAddress.get_key(address)
        path = os.path.join(self.storage_loc, address_key)
        if os.path.isfile(path):
            return os.path.getsize(path)
        else:
            # From https://stackoverflow.com/questions/1392413/calculating-a-directorys-size-using-python
            object_size = 0
            for dirpath, dirnames, filenames in os.walk(path):
                for f in filenames:
                    fp = os.path.join(dirpath, f)
                    if not os.path.islink(fp):
                        object_size += os.path.getsize(fp)
            return object_size

    def delete_object(self, address: str, kind: str = "data") -> bool:
        """Delete an object from disk datastore.

        Parameters
        ----------
        address : str
            Address of the object.
        kind : str, optional
            Type of object ('data', 'model', 'dir'), by default 'data'.

        Returns
        -------
        bool
            True if deletion was successful.
        """
        key = os.path.join(self.storage_loc, DeepchemAddress.get_key(address))
        if kind == 'data':
            card_key = key + '.cdc'
        elif kind == 'model':
            card_key = key + '.cmc'
        elif kind == 'dir':
            card_key = None
        if os.path.isfile(key):
            os.remove(key)
        else:
            shutil.rmtree(key)
        if card_key:
            os.remove(card_key)
        return True

    def download_object(self, address: str, filename: Union[str, Path, None] = None) -> None:
        """
        Downloads a object from disk datastore

        Parameters
        ----------
        address: str
          DeepchemAddress of the object

        Returns
        -------
        None

        Note
        ----
        Dataset download is not meaningful in disk datastore since the dataset
        already exists in the disk. Hence, we make a copy of the file at the location
        specified by `filename`.
        """
        key = DeepchemAddress.get_key(address)
        path = os.path.join(self.storage_loc, key)
        if not filename:
            raise ValueError("filename should be set")
        if os.path.isfile(path):
            shutil.copyfile(path, filename)
        elif os.path.isdir(path):
            shutil.copytree(path, filename, dirs_exist_ok=True)
        return None

    def get_object_size(self, address: str) -> int:
        """
        Returns size of the object

        Parameters
        ----------
        address: str
          DeepchemAddress of the object

        Returns
        -------
        object_size: int
          Size of the object
        """
        address_key = DeepchemAddress.get_key(address)
        object_path = os.path.join(self.storage_loc, address_key)
        if os.path.isfile(object_path):
            return os.path.getsize(object_path)
        else:
            object_size = 0
            for dirpath, dirnames, filenames in os.walk(object_path):
                for f in filenames:
                    fp = os.path.join(dirpath, f)
                    if not os.path.islink(fp):
                        object_size += os.path.getsize(fp)
            return object_size

    def move_object(self, source_address: str, dest_address: str, dest_datastore: DiskDataStore) -> None:
        """
        Move an object from one location to another

        Parameters
        ----------
        source_address: str
            DeepchemAddress of the object to be moved
        dest_address: str
            DeepchemAddress of the destination
        dest_datastore: DiskDataStore
            DiskDataStore object of the destination

        Returns
        -------
        None
        """

        # Get the source key and destination key of the object and the card
        source_key = os.path.join(self.storage_loc, DeepchemAddress.get_key(source_address))
        for kind in KIND_LIST:
            if source_key.endswith(kind['extension']):
                raise ValueError("Cannot move a card")

        # Determine the kind of the object
        kind_matched: Optional[dict] = None
        for kind in KIND_LIST:
            if os.path.exists(source_key + kind['extension']):
                kind_matched = kind
                break
        if not kind_matched and not os.path.isdir(source_key):
            raise ValueError("Source does not have a card")

        dest_key = os.path.join(dest_datastore.storage_loc, DeepchemAddress.get_key(dest_address))
        dest_dir = os.path.dirname(dest_key)
        if dest_key.strip('/') != dest_dir.strip('/') and not os.path.exists(dest_dir):
            os.makedirs(dest_dir)

        for kind in KIND_LIST:
            if dest_key.endswith(kind['extension']):
                raise ValueError("Destination cannot be a card")
        dest_card_key = (dest_key + kind_matched['extension']) if kind_matched else None

        # dest_key and dest_card_key should not exist
        if os.path.exists(dest_key) and (not kind_matched or (dest_card_key and os.path.exists(dest_card_key))):
            raise FileExistsError("Destination object already exists")

        # Check if source is a directory
        if os.path.isdir(source_key):
            # Copy the directory to the destination
            shutil.copytree(source_key, dest_key)
            if kind_matched and dest_card_key:
                shutil.copyfile(source_key + kind_matched['extension'], dest_card_key)
        else:
            # Copy the object to the destination
            shutil.copyfile(source_key, dest_key)

        # Update the address in the card and write it to the destination
        if kind_matched:
            card = self.get_card(source_address, kind=kind_matched['name'])
            if not card:
                raise ValueError("Card not found")
            card.address = dest_address
            if dest_card_key is not None:
                with open(dest_card_key, 'wb') as fp:
                    fp.write(bytes(card))
            else:
                raise ValueError("Destination card key could not be determined.")

        # Delete the object and the card from the source
        self.delete_object(source_address, kind=kind_matched['name'] if kind_matched else 'dir')

    def copy_object(self, source_address: str, dest_address: str, dest_datastore: DiskDataStore) -> None:
        """
        Copy an object from one location to another

        Parameters
        ----------
        source_address: str
            DeepchemAddress of the object to be copied
        dest_address: str
            DeepchemAddress of the destination
        dest_datastore: DiskDataStore
            DiskDataStore object of the destination

        Returns
        -------
        None
        """

        # Get the source key and destination key of the object and the card
        source_key = os.path.join(self.storage_loc, DeepchemAddress.get_key(source_address))
        for kind in KIND_LIST:
            if source_key.endswith(kind['extension']):
                raise ValueError("Cannot move a card")

        # Determine the kind of the object
        kind_matched: Optional[dict] = None
        for kind in KIND_LIST:
            if os.path.exists(source_key + kind['extension']):
                kind_matched = kind
                break
        if not kind_matched and not os.path.isdir(source_key):
            raise ValueError("Source does not have a card")

        dest_key = os.path.join(dest_datastore.storage_loc, DeepchemAddress.get_key(dest_address))
        dest_dir = os.path.dirname(dest_key)
        if dest_key.strip('/') != dest_dir.strip('/') and not os.path.exists(dest_dir):
            os.makedirs(dest_dir)

        for kind in KIND_LIST:
            if dest_key.endswith(kind['extension']):
                raise ValueError("Destination cannot be a card")
        dest_card_key = (dest_key + kind_matched['extension']) if kind_matched else None

        # dest_key and dest_card_key should not exist
        if os.path.exists(dest_key) and (dest_card_key and os.path.exists(dest_card_key)):
            raise FileExistsError("Destination object already exists")

        # Check if source is a directory
        if os.path.isdir(source_key):
            # Copy the directory to the destination
            shutil.copytree(source_key, dest_key)
            if kind_matched and dest_card_key:
                shutil.copyfile(source_key + kind_matched['extension'], dest_card_key)
        else:
            # Copy the object to the destination
            shutil.copyfile(source_key, dest_key)

        # Update the address in the card and write it to the destination
        if kind_matched:
            card = self.get_card(source_address, kind=kind_matched['name'])
            if not card:
                raise ValueError("Card not found")
            card.address = dest_address
            if dest_card_key is not None:
                with open(dest_card_key, 'wb') as fp:
                    fp.write(bytes(card))
            else:
                raise ValueError("Destination card key could not be determined.")

    def exists(self, address: str) -> bool:
        """
        Check if an object exists in the datastore

        Parameters
        ----------
        address: str
          DeepchemAddress of the object

        Returns
        -------
        bool
          True if the object exists, False otherwise
        """
        key = os.path.join(self.storage_loc, DeepchemAddress.get_key(address))
        return os.path.exists(key)

    def __repr__(self) -> str:
        """Return objects in the DiskDataStore.

        Returns
        -------
        str
            String representation of all objects in the datastore.
        """
        # TODO A pretty print of objects. Ref: https://docs.python.org/3/library/pprint.html
        all_objects = self._get_datastore_objects(self.storage_loc)
        objects = []
        for i, object_ in enumerate(all_objects):
            objects.append('deepchem://' + self.storage_loc + '/' + object_)
        return '\n'.join(objects)

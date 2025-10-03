import ast
import multiprocessing as mp
import os
import re
import tempfile
from typing import Dict, Iterable, List, Optional, Set, Union

import deepchem as dc
import pandas as pd
from rdkit import Chem

from deepchem_server.core import config
from deepchem_server.core.address import DeepchemAddress
from deepchem_server.core.cards import DataCard


featurizer_map = {
    "ecfp": dc.feat.CircularFingerprint,
    "graphconv": dc.feat.ConvMolFeaturizer,
    "weave": dc.feat.WeaveFeaturizer,
    "molgraphconv": dc.feat.MolGraphConvFeaturizer,
    "dummy": dc.feat.DummyFeaturizer,
    "grover": dc.feat.GroverFeaturizer,
    "rdkitconformer": dc.feat.RDKitConformerFeaturizer,
    "dmpnn": dc.feat.DMPNNFeaturizer,
}


def split_dataset(dataset_path: str, file_type: str, n_partition: int, available_checkpoints: List[int]) -> List[str]:
    """Split the dataset into n partitions.

    Parameters
    ----------
    dataset_path : str
        The path to the dataset.
    file_type : str
        The type of the dataset (e.g., 'csv', 'sdf').
    n_partition : int
        The number of partitions to split the dataset into.
    available_checkpoints : list of int
        List of checkpoint partition IDs that are already available.

    Returns
    -------
    list of str
        The list of file paths of the partitioned datasets.

    Raises
    ------
    NotImplementedError
        If the file type is not supported for featurization.
    """
    basedir = os.path.dirname(dataset_path)
    datasets = []
    if file_type == 'csv':
        df = pd.read_csv(dataset_path)
        part_size = df.shape[0] // n_partition
        overflow = df.shape[0] % n_partition  # remainder of the datapoints after n equal partitions
        for i in range(n_partition):
            if i in available_checkpoints:
                continue
            start = i * part_size
            end = (i + 1) * part_size
            if i == n_partition - 1:
                end += overflow  # adds overflow datapoints to the last partition
            df_subset = df.iloc[start:end]
            dest_dir = os.path.join(basedir, f'part{i}')
            os.makedirs(dest_dir, exist_ok=True)
            csv_path = os.path.join(dest_dir, f'part{i}.csv')
            datasets.append(csv_path)
            df_subset.to_csv(csv_path, index=False)
    elif file_type == 'sdf':
        suppl = Chem.SDMolSupplier(dataset_path)
        mol_l = []
        for mol in suppl:
            mol_l.append(mol)

        n_rows = len(mol_l)
        part_size = n_rows // n_partition
        overflow = n_rows % n_partition  # remainder of the datapoints after n equal partitions
        for i in range(n_partition):
            if i in available_checkpoints:
                continue
            start = i * part_size
            end = (i + 1) * part_size
            if i == n_partition - 1:
                end += overflow  # adds overflow datapoints to the last partition
            dest_dir = os.path.join(basedir, f'part{i}')
            os.makedirs(dest_dir, exist_ok=True)
            sdf_path = os.path.join(dest_dir, f'part{i}.sdf')
            datasets.append(sdf_path)
            writer = Chem.SDWriter(sdf_path)
            for mol in mol_l[start:end]:
                writer.write(mol)
    else:
        raise NotImplementedError(f"Featurization of {dataset_path} not supported.")
    return datasets


def featurize_part(
    main_dataset_address: str,
    dataset_path: str,
    file_type: str,
    featurizer: dc.feat.Featurizer,
    dataset_column: str,
    label_column: Optional[str],
    checkpoint_output_key: str,
    nproc: int,
) -> None:
    """Featurize a part of the dataset.

    Parameters
    ----------
    main_dataset_address : str
        Address of the main dataset being featurized.
    dataset_path : str
        The path to the dataset partition to featurize.
    file_type : str
        The type of the dataset (e.g., 'csv', 'sdf').
    featurizer : dc.feat.Featurizer
        The featurizer to use.
    dataset_column : str
        The column containing the input for featurizer.
    label_column : str, optional
        The target column in case this dataset is going to be used for
        training purposes.
    checkpoint_output_key : str
        The output key for checkpoint '.partial' folder.
    nproc : int
        The total number of partitions being processed.

    Returns
    -------
    None

    Raises
    ------
    ValueError
        If datastore is not set.
    NotImplementedError
        If the file type is not supported for featurization.
    """
    datastore = config.get_datastore()
    if datastore is None:
        raise ValueError("Datastore not set")

    dest_dir = os.path.dirname(dataset_path)

    if file_type == 'csv':
        if label_column is not None:
            loader = dc.data.CSVLoader(tasks=[label_column], feature_field=dataset_column, featurizer=featurizer)
        else:
            loader = dc.data.CSVLoader(tasks=[], feature_field=dataset_column, featurizer=featurizer)
        dataset = loader.create_dataset(dataset_path)
    elif file_type == 'sdf':
        if label_column is not None:
            loader = dc.data.SDFLoader(tasks=[label_column], featurizer=featurizer, sanitize=True)
        else:
            loader = dc.data.SDFLoader(tasks=[], featurizer=featurizer, sanitize=True)
        dataset = loader.create_dataset(dataset_path)
    else:
        raise NotImplementedError(f"Featurization of '{file_type}' not supported.")
    dataset.move(dest_dir)
    checkpoint_id = dataset_path.split('/')[-1].split('.')[0][-1]
    card = DataCard(address='',
                    file_type='dir',
                    featurizer=featurizer,
                    data_type=type(dataset).__name__,
                    checkpoint_id=checkpoint_id,
                    n_core=nproc,
                    parent=main_dataset_address,
                    description=f"featurized partition of \
            {main_dataset_address} : {checkpoint_id} of {nproc - 1}")

    datastore.upload_data_from_memory(dataset,
                                      checkpoint_output_key + f'/_checkpoint/part_{checkpoint_id}_of_{nproc - 1}', card)


def featurize_multi_core(
    main_dataset_address: str,
    raw_dataset_path: str,
    file_type: str,
    feat: dc.feat.Featurizer,
    dataset_column: str,
    label_column: Optional[str],
    basedir: str,
    nproc: int,
    checkpoint_output_key: str,
    available_checkpoints: List[int],
) -> Iterable[Union[List, str]]:
    """Featurize the dataset in parallel.

    Parameters
    ----------
    main_dataset_address : str
        Address of the main dataset being featurized.
    raw_dataset_path : str
        The path to the raw dataset.
    file_type : str
        The type of the dataset (e.g., 'csv', 'sdf').
    feat : dc.feat.Featurizer
        The featurizer to use.
    dataset_column : str
        The column containing the input for featurizer.
    label_column : str, optional
        The target column in case this dataset is going to be used for
        training purposes.
    basedir : str
        The base directory where the dataset is stored.
    nproc : int
        The number of partitions to split the dataset into.
    checkpoint_output_key : str
        The output key for checkpoint '.partial' folder.
    available_checkpoints : list of int
        The list of checkpoint ids already completed in the previous run (if any).

    Returns
    -------
    list
        A list containing [datasets, merge_dir] where datasets is a list of
        DiskDataset objects and merge_dir is the directory path for merging.
    """
    dataset_paths = split_dataset(raw_dataset_path, file_type, nproc, available_checkpoints)
    processes = []

    for dataset_path in dataset_paths:
        p = mp.Process(target=featurize_part,
                       args=(main_dataset_address, dataset_path, file_type, feat, dataset_column, label_column,
                             checkpoint_output_key, nproc))
        processes.append(p)
        p.start()
    for p in processes:
        p.join()

    merge_dir = os.path.join(basedir, 'merged')
    os.makedirs(merge_dir, exist_ok=True)

    datasets = []
    for i in range(nproc):
        dataset_dir = os.path.join(basedir, f'part{i}')
        datasets.append(dc.data.DiskDataset(data_dir=dataset_dir))

    return [datasets, merge_dir]


def featurize(
    dataset_address: str,
    featurizer: str,
    output: str,
    dataset_column: str,
    feat_kwargs: Dict = dict(),
    label_column: Optional[str] = None,
    n_core: Optional[int] = None,
    single_core_threshold: Optional[int] = 250,
) -> Optional[str]:
    """Featurize the dataset at given address with specified featurizer.

    Writes output to datastore. If the compute node has more than 1 CPU core
    then the featurization is done by splitting the dataset into parts of equal
    size and featurizing each part in parallel. The featurized parts are then
    merged into a single dataset and written to the datastore. The number of
    parts is equal to the number of cores available on the machine. If the
    compute node has only 1 CPU core then the featurization will be done in a
    single process.

    Restart support:
    The featurize primitive saves a `(output).partial` folder where the
    checkpoints are saved until completion. To resume a failed featurize
    execution, the featurize primitive can be rerun with the same arguments and
    the checkpoints will be restored from the `(output).partial` folder and the
    folder is deleted once the featurization process is complete.

    Note: The restart fails if the n_core < n_core used before restart.
    Additionally, the checkpoints must belong to the same dataset address as the
    initial run, otherwise, they will not be considered for the restart.

    Parameters
    ----------
    dataset_address : str
        The deepchem address of the dataset to featurize.
    featurizer : str
        Has to be a featurizer string in mappings.
    output : str
        The name of output featurized dataset in your workspace.
    dataset_column : str
        Column containing the input for featurizer.
    feat_kwargs : dict, optional
        Keyword arguments to pass to featurizer on initialization, by default {}.
    label_column : str, optional
        The target column in case this dataset is going to be used for
        training purposes.
    n_core : int, optional
        The number of cores to use for featurization.
    single_core_threshold : int, optional
        The threshold size of the dataset size in megabytes above which
        multicore featurization will be used, by default 250.

    Returns
    -------
    str
        Deepchem address of the featurized dataset.

    Raises
    ------
    ValueError
        If featurizer is not recognized, if input column is not specified for
        CSV files, or if datastore is not set.
    NotImplementedError
        If the dataset format is not supported for featurization.
    """
    # TODO Allow a list of label column for multitask learning
    # TODO: Handle parsing of dictionary via parser
    featurizer = featurizer.lower()
    if featurizer not in featurizer_map:
        raise ValueError(f"Featurizer not recognized.\nAvailable featurizers: {featurizer_map}")
    if dataset_address.endswith('csv'):
        if dataset_column == 'None' or dataset_column is None:
            raise ValueError("Please specify input column.")

    if isinstance(feat_kwargs, str):
        feat_kwargs = ast.literal_eval(feat_kwargs)
    if label_column == 'None':
        label_column = None
    feat_kwargs_restore = {}
    if feat_kwargs:
        if "features_generator" in feat_kwargs:
            feat_generator = feat_kwargs["features_generator"]
            feat_kwargs_restore["features_generator"] = feat_generator
            feat_kwargs["features_generator"] = featurizer_map[feat_generator]()
        feat = featurizer_map[featurizer](**feat_kwargs)
    else:
        feat = featurizer_map[featurizer]()
    assert dataset_address.endswith('csv') or dataset_address.endswith('sdf')

    datastore = config.get_datastore()
    if datastore is None:
        raise ValueError("Datastore not set")

    output_key = DeepchemAddress.get_key(output)
    checkpoint_output_key = output_key + ".partial"
    tempdir = tempfile.TemporaryDirectory()
    basedir = os.path.join(tempdir.name)

    if datastore.exists(output_key):
        raise FileExistsError(
            f"Output address {output_key} already exists in datastore. Please choose a different output name.")

    # check if _checkpoint/ folder exists in given output folder in datastore
    available_checkpoints = []
    _storage_loc = datastore.storage_loc.rstrip("/")
    pattern = re.compile(fr"{_storage_loc}/{checkpoint_output_key}/_checkpoint/part_\d+_of_\d+\.cdc$")
    n_core_set: Set[int] = set()
    for item in datastore._get_datastore_objects(_storage_loc):
        match = pattern.search(item)
        if match:
            card = datastore.get(item)
            if card and hasattr(card, 'parent') and card.parent == dataset_address:
                n_core_set.add(card.n_core)
                available_checkpoints.append(int(card.checkpoint_id))
                chkpt_tmp_path: str = os.path.join(tempdir.name, f'part{int(card.checkpoint_id)}')
                chkpt_address = item[:-4]  # removes .cdc
                datastore.download_object(chkpt_address, chkpt_tmp_path)

    if len(n_core_set) == 1:
        n_core = list(n_core_set)[0]
        if n_core > os.cpu_count():  # type: ignore
            raise Exception(
                f"Current job config is insufficient to restart the job as it requires atleast {n_core // 2} vcpus")
    elif len(n_core_set) > 1:
        raise Exception("Checkpoints found with more than one partition type")

    if n_core is None:
        nproc = os.cpu_count()
    else:
        nproc = n_core

    if dataset_address.endswith('csv'):
        raw_dataset_path = os.path.join(tempdir.name, 'temp.csv')
        dataset_size = datastore.get_file_size(dataset_address)

        datastore.download_object(dataset_address, raw_dataset_path)
        file_size = dataset_size / (1000 * 1000)

        if nproc and nproc > 1 and file_size and single_core_threshold and file_size > single_core_threshold:
            datasets, merge_dir = featurize_multi_core(dataset_address, raw_dataset_path, 'csv', feat, dataset_column,
                                                       label_column, basedir, nproc, checkpoint_output_key,
                                                       available_checkpoints)
            dataset = dc.data.DiskDataset.merge(datasets, merge_dir=merge_dir)
        else:
            if label_column is not None:
                loader = dc.data.CSVLoader(tasks=[label_column], feature_field=dataset_column, featurizer=feat)
            else:
                loader = dc.data.CSVLoader(tasks=[], feature_field=dataset_column, featurizer=feat)
            dataset = loader.create_dataset(raw_dataset_path)

    elif dataset_address.endswith('sdf'):
        raw_dataset_path = os.path.join(tempdir.name, 'temp.sdf')
        dataset_size = datastore.get_file_size(dataset_address)
        datastore.download_object(dataset_address, raw_dataset_path)
        file_size = dataset_size / (1024 * 1024)

        if nproc and nproc > 1 and file_size and single_core_threshold and file_size > single_core_threshold:
            datasets, merge_dir = featurize_multi_core(dataset_address, raw_dataset_path, 'sdf', feat, dataset_column,
                                                       label_column, basedir, nproc, checkpoint_output_key,
                                                       available_checkpoints)
            dataset = dc.data.DiskDataset.merge(datasets, merge_dir=merge_dir)
        else:
            if label_column is not None:
                loader = dc.data.SDFLoader(tasks=[label_column], featurizer=feat, sanitize=True)
            else:
                loader = dc.data.SDFLoader(tasks=[], featurizer=feat, sanitize=True)
            dataset = loader.create_dataset(raw_dataset_path)

    else:
        raise NotImplementedError(f"Featurization of {dataset_address} not supported.")

    for key, value in feat_kwargs_restore.items():
        feat_kwargs[key] = value

    card = DataCard(address='',
                    file_type='dir',
                    featurizer=featurizer,
                    data_type=type(dataset).__name__,
                    feat_kwargs=feat_kwargs)

    featurized_address = datastore.upload_data_from_memory(dataset, output_key, card)

    if checkpoint_output_key + '/' in datastore.list_data():
        datastore.delete_object(address=_storage_loc + "/" + checkpoint_output_key, kind='dir')
    return featurized_address

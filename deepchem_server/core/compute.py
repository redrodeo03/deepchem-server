from typing import Any, Dict

import deepchem_server.core as core


program_map = {
    "featurize": core.featurize,
    "train": core.train,
    "evaluate": core.model_evaluator,
    "infer": core.infer,
    "train_valid_test_split": core.train_valid_test_split,,
    "generate_pose": core.generate_pose,
}


class ComputeWorkflow:
    """A Compute Workflow is a workflow that runs on Deepchem Server.

    Parameters
    ----------
    program : Dict
        A dictionary containing program configuration including 'program_name'
        and other parameters required by the specific program.

    Examples
    --------
    >>> program = {
    ...     'program_name': 'featurize',
    ...     'dataset_address': 'deepchem://profile_name/project_name/data.csv',
    ...     'featurizer': 'ecfp',
    ...     'output': 'test_output',
    ...     'dataset_column': 'smiles',
    ...     'feat_kwargs': {'size': 1024},
    ...     'label_column': 'y',
    ... }
    >>> workflow = ComputeWorkflow(program)
    """

    def __init__(self, program: Dict) -> None:
        """Initialize ComputeWorkflow.

        Parameters
        ----------
        program : Dict
            A dictionary containing program configuration.
        """
        self.program: Dict = program

    def execute(self) -> Any:
        """Run the program based on the 'program_name' in the program dictionary.

        Returns
        -------
        Any
            The output of the executed program.

        Raises
        ------
        ValueError
            If 'program_name' is not found in program or if the program_name
            is not available in the program map.
        """
        if 'program_name' not in self.program:
            raise ValueError("program_name not found in program")
        program_name: str = self.program['program_name']
        params: Dict = {key: value for key, value in self.program.items() if key != 'program_name'}
        if program_name not in program_map:
            raise ValueError(f"{program_name} not in available programs")

        output: Any = program_map[program_name](**params)  # type: ignore
        return output

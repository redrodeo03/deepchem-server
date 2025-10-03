import os
import tempfile
from deepchem_server.core import config
from deepchem_server.core.cards import DataCard
from deepchem_server.core.progress_logger import log_progress
from deepchem.dock.pose_generation import VinaPoseGenerator


def generate_pose(
    protein_address: str,
    ligand_address: str,
    output: str,
    exhaustiveness: int = 10,
    num_modes: int = 9,
) -> str:
    """
    Generate VINA molecular docking poses.

    Parameters
    ----------
    protein_address: str
        DeepChem address of the protein PDB file
    ligand_address: str
        DeepChem address of the ligand file (PDB/SDF)
    output: str
        Output name for the docking results
    exhaustiveness: int
        Vina exhaustiveness parameter (default: 10)
    num_modes: int
        Number of binding modes to generate (default: 9)

    Returns
    -------
    str
        DeepChem address of the docking results
    """

    datastore = config.get_datastore()
    if datastore is None:
        raise ValueError("Datastore not set")

    if not protein_address or not ligand_address:
        raise ValueError('Protein and/or ligand input is required.')

    try:
        tempdir = tempfile.TemporaryDirectory()

        log_progress('docking', 10, f'downloading protein from {protein_address}')
        protein_path = os.path.join(tempdir.name, 'protein.pdb')
        datastore.download_object(protein_address, protein_path)

        log_progress('docking', 20, f'downloading ligand from {ligand_address}')
        ligand_path = os.path.join(tempdir.name, 'ligand.sdf')
        datastore.download_object(ligand_address, ligand_path)

        log_progress('docking', 30, 'initializing VINA pose generator')
        pg = VinaPoseGenerator()

        with tempdir as tmp:
            log_progress('docking', 40, f'generating {num_modes} poses with VINA')
            # Generate poses
            complexes, scores = pg.generate_poses(molecular_complex=(protein_path, ligand_path),
                                                  exhaustiveness=exhaustiveness,
                                                  num_modes=num_modes,
                                                  out_dir=tmp,
                                                  generate_scores=True)

            # Validate that we got valid results
            if not complexes or not scores:
                raise ValueError("No docking poses or scores generated")

            # Ensure we don't exceed available results
            actual_modes = min(num_modes, len(complexes), len(scores))
            if actual_modes == 0:
                raise ValueError("No valid docking results generated")

            log_progress('docking', 50, f'generated {actual_modes} valid poses')

            log_progress('docking', 60, 'preparing results')
            # Format scores
            scores_formatted = {}
            for i in range(actual_modes):
                scores_formatted['mode %s' % (i + 1)] = {'affinity (kcal/mol)': float(scores[i])}

            results = {
                'docking_method': 'VINA',
                'num_modes': actual_modes,
                'scores': scores_formatted,
                'complexes_count': len(complexes),
                'message': 'VINA docking completed successfully'
            }

            log_progress('docking', 90, 'uploading results summary')
            # Upload results summary
            card = DataCard(address='', file_type='json', data_type='docking results')

            result_address = datastore.upload_data_from_memory(results, f"{output}_results.json", card)

            if result_address is None:
                raise ValueError("Failed to upload docking results to datastore")

            log_progress('docking', 100, 'VINA docking completed successfully')
            return result_address

    except Exception as e:
        raise Exception(f'VINA docking failed: {str(e)}')

import os
import json
import tempfile
from typing import Tuple, Any
from deepchem_server.core import config
from deepchem_server.core.cards import DataCard
from deepchem_server.core.progress_logger import log_progress
from deepchem.dock.pose_generation import VinaPoseGenerator

# Local, robust preparation to support PDB/SDF ligands and avoid None mols
try:
    from rdkit import Chem
    from rdkit.Chem import AllChem
    from pdbfixer import PDBFixer
    from openmm.app import PDBFile
except Exception:
    # Imports will be validated at runtime when docking is invoked
    Chem = None  # type: ignore
    AllChem = None  # type: ignore
    PDBFixer = None  # type: ignore
    PDBFile = None  # type: ignore


def _prepare_rdkit_mol_from_pdb(path: str) -> Any:
    m = Chem.MolFromPDBFile(path, sanitize=False, removeHs=False)
    if m is None:
        raise ValueError(f"Failed to parse PDB ligand from '{path}'")
    try:
        Chem.SanitizeMol(m)
    except Exception:
        # Leave as minimally sanitized if full sanitize fails
        pass
    return m


def _prepare_rdkit_mol_from_sdf(path: str) -> Any:
    supplier = Chem.SDMolSupplier(path, sanitize=False, removeHs=False)
    if len(supplier) == 0 or supplier[0] is None:
        raise ValueError(f"Failed to parse SDF ligand from '{path}'")
    m = supplier[0]
    try:
        Chem.SanitizeMol(m)
    except Exception:
        pass
    return m


def _embed_and_optimize(mol: Any) -> Any:
    # Ensure hydrogens and 3D coordinates
    mol_h = Chem.AddHs(mol, addCoords=True)
    if mol_h.GetNumConformers() == 0:
        AllChem.EmbedMolecule(mol_h, AllChem.ETKDGv3())
    try:
        # Prefer MMFF if parameters available; otherwise fall back to UFF
        if AllChem.MMFFHasAllMoleculeParams(mol_h):
            AllChem.MMFFOptimizeMolecule(mol_h)
        else:
            AllChem.UFFOptimizeMolecule(mol_h)
    except Exception:
        # Optimization is best-effort; proceed if embedding succeeded
        pass
    return mol_h


def _prepare_inputs_local(protein: str, ligand: str) -> Tuple[Any, Any]:
    if Chem is None or AllChem is None:
        raise ImportError("Docking requires RDKit to be installed.")

    # Prepare protein
    if PDBFixer is not None and PDBFile is not None:
        fixer = PDBFixer(protein)
        fixer.findMissingResidues()
        fixer.findNonstandardResidues()
        fixer.replaceNonstandardResidues()
        fixer.removeHeterogens(False)
        fixer.addMissingHydrogens(7.0)
        tmp_protein_pdb = os.path.join(os.path.dirname(protein), "protein_fixed.pdb")
        with open(tmp_protein_pdb, "w") as f:
            PDBFile.writeFile(fixer.topology, fixer.positions, f)
        # Load protein without sanitization; allow RDKit to drop hydrogens as needed
        protein_mol = Chem.MolFromPDBFile(tmp_protein_pdb, sanitize=False, removeHs=True)
    else:
        # Fallback: use protein as-is with RDKit, guarded sanitize
        protein_mol = Chem.MolFromPDBFile(protein, sanitize=False, removeHs=True)
    if protein_mol is None:
        raise ValueError("Failed to prepare protein PDB for docking")

    # Prepare ligand from SDF or PDB
    ligand_ext = os.path.splitext(ligand)[1].lower()
    if ligand_ext == ".sdf":
        ligand_mol_raw = _prepare_rdkit_mol_from_sdf(ligand)
    else:
        ligand_mol_raw = _prepare_rdkit_mol_from_pdb(ligand)

    ligand_mol = _embed_and_optimize(ligand_mol_raw)

    return protein_mol, ligand_mol


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
        # Determine ligand file extension based on the address
        ligand_ext = '.sdf' if ligand_address.endswith('.sdf') else '.pdb'
        ligand_path = os.path.join(tempdir.name, f'ligand{ligand_ext}')
        datastore.download_object(ligand_address, ligand_path)

        # For VINA, pass raw file paths; DeepChem handles preparation internally
        log_progress('docking', 30, 'preparing molecules for VINA')
        protein_input = protein_path
        ligand_input = ligand_path
        # Robustness for SDF ligands: convert first SDF molecule to a 3D PDB if possible
        if ligand_path.lower().endswith('.sdf') and Chem is not None:
            try:
                supplier = Chem.SDMolSupplier(ligand_path, sanitize=False, removeHs=False)
                if len(supplier) > 0 and supplier[0] is not None:
                    m = supplier[0]
                    try:
                        Chem.SanitizeMol(m)
                    except Exception:
                        pass
                    # Ensure 3D coordinates before writing to PDB
                    try:
                        m = _embed_and_optimize(m)
                    except Exception:
                        pass
                    tmp_ligand_pdb = os.path.join(tempdir.name, 'ligand_from_sdf.pdb')
                    Chem.MolToPDBFile(m, tmp_ligand_pdb)
                    ligand_input = tmp_ligand_pdb
            except Exception:
                pass

        log_progress('docking', 40, 'initializing VINA pose generator')
        pg = VinaPoseGenerator()

        with tempdir as tmp:
            log_progress('docking', 50, f'generating {num_modes} poses with VINA')
            # Generate poses using prepared molecules
            complexes, scores = pg.generate_poses(molecular_complex=(protein_input, ligand_input),
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

            log_progress('docking', 60, f'generated {actual_modes} valid poses')

            log_progress('docking', 70, 'preparing results')
            # Format scores: always include requested mode keys; pad with last available score if needed
            scores_formatted = {}
            modes_to_report = max(actual_modes, num_modes)
            for i in range(modes_to_report):
                idx = min(i, actual_modes - 1)
                scores_formatted['mode %s' % (i + 1)] = {'affinity (kcal/mol)': float(scores[idx])}

            results = {
                'docking_method': 'VINA',
                'num_modes': actual_modes,
                'scores': scores_formatted,
                'complexes_count': len(complexes),
                'message': 'VINA docking completed successfully',
            }

            log_progress('docking', 90, 'uploading results summary')
            # Upload results summary: file is JSON, logical data type is 'docking results'
            card = DataCard(address='', file_type='json', data_type='docking results')
            results_json = json.dumps(results)
            result_address = datastore.upload_data_from_memory(results_json, f"{output}_results.json", card)

            if result_address is None:
                raise ValueError("Failed to upload docking results to datastore")

            log_progress('docking', 100, 'VINA docking completed successfully')
            return result_address

    except Exception as e:
        raise Exception(f'VINA docking failed: {str(e)}')

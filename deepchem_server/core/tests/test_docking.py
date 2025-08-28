from deepchem_server.core.cards import DataCard
from deepchem_server.core import config, generate_pose
import os
import json
import tempfile
import shutil
import pytest


def test_generate_pose_basic_functionality(disk_datastore):
    """Test basic VINA pose generation functionality."""
    config.set_datastore(disk_datastore)
    
    # Create test protein and ligand files
    current_dir = os.path.dirname(os.path.abspath(__file__))
    pdb_test_file = os.path.join(current_dir, "assets/cleaned_3cyx.pdb")
    ligand_test_file = os.path.join(current_dir, "assets/ligand_3cyx.pdb")
    
    # Upload test files to datastore
    card_protein = DataCard(address='', file_type='pdb', data_type='text/plain')
    card_ligand = DataCard(address='', file_type='pdb', data_type='text/plain')
    
    pdb_address = disk_datastore.upload_data("protein_test.pdb", pdb_test_file, card_protein)
    ligand_address = disk_datastore.upload_data("ligand_test.pdb", ligand_test_file, card_ligand)
    
    # Test basic pose generation
    result_address = generate_pose(
        protein_address=pdb_address,
        ligand_address=ligand_address,
        output='test_output',
        exhaustiveness=1,
        num_modes=1
    )
    
    # Verify result structure
    assert result_address.startswith('deepchem://')
    assert result_address.endswith('_results.json')
    
    # Download and verify results
    results_data = disk_datastore.get(result_address)
    results = json.loads(results_data)
    
    # Check basic structure
    assert 'docking_method' in results
    assert 'num_modes' in results
    assert 'scores' in results
    assert 'complexes_count' in results
    assert 'message' in results
    
    # Check specific values
    assert results['docking_method'] == 'VINA'
    assert results['num_modes'] >= 0
    assert results['complexes_count'] >= 0
    assert 'VINA docking completed successfully' in results['message']


def test_generate_pose_multiple_modes(disk_datastore):
    """Test VINA pose generation with multiple modes."""
    config.set_datastore(disk_datastore)
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    pdb_test_file = os.path.join(current_dir, "assets/cleaned_3cyx.pdb")
    ligand_test_file = os.path.join(current_dir, "assets/ligand_3cyx.pdb")
    
    card_protein = DataCard(address='', file_type='pdb', data_type='text/plain')
    card_ligand = DataCard(address='', file_type='pdb', data_type='text/plain')
    
    pdb_address = disk_datastore.upload_data("protein_test.pdb", pdb_test_file, card_protein)
    ligand_address = disk_datastore.upload_data("ligand_test.pdb", ligand_test_file, card_ligand)
    
    # Test with multiple modes
    result_address = generate_pose(
        protein_address=pdb_address,
        ligand_address=ligand_address,
        output='test_output',
        exhaustiveness=1,
        num_modes=3
    )
    
    # Download and verify results
    results_data = disk_datastore.get(result_address)
    results = json.loads(results_data)
    
    # Check that we have scores for multiple modes
    assert 'scores' in results
    scores = results['scores']
    
    # Check that scores follow the expected format: 'mode %s' % (i + 1)
    mode_keys = list(scores.keys())
    assert len(mode_keys) > 0
    
    # Verify each mode has the expected structure
    for mode_key in mode_keys:
        assert mode_key.startswith('mode ')
        assert 'affinity (kcal/mol)' in scores[mode_key]
        # Scores should be numeric
        assert isinstance(scores[mode_key]['affinity (kcal/mol)'], (int, float))


def test_generate_pose_sdf_ligands(disk_datastore):
    """Test VINA pose generation with SDF ligand files."""
    config.set_datastore(disk_datastore)
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    pdb_test_file = os.path.join(current_dir, "assets/cleaned_3cyx.pdb")
    ligand_sdf_file = os.path.join(current_dir, "assets/ligand_3cyx.sdf")
    
    card_protein = DataCard(address='', file_type='pdb', data_type='text/plain')
    card_ligand = DataCard(address='', file_type='sdf', data_type='text/plain')
    
    pdb_address = disk_datastore.upload_data("protein_test.pdb", pdb_test_file, card_protein)
    ligand_address = disk_datastore.upload_data("ligand_test.sdf", ligand_sdf_file, card_ligand)
    
    # Test with SDF ligand
    result_address = generate_pose(
        protein_address=pdb_address,
        ligand_address=ligand_address,
        output='test_output_sdf',
        exhaustiveness=1,
        num_modes=1
    )
    
    # Verify result
    assert result_address.startswith('deepchem://')
    assert result_address.endswith('_results.json')
    
    # Download and verify results
    results_data = disk_datastore.get(result_address)
    results = json.loads(results_data)
    
    assert results['docking_method'] == 'VINA'
    assert results['num_modes'] >= 0


def test_generate_pose_exhaustiveness_parameter(disk_datastore):
    """Test VINA pose generation with different exhaustiveness values."""
    config.set_datastore(disk_datastore)
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    pdb_test_file = os.path.join(current_dir, "assets/cleaned_3cyx.pdb")
    ligand_test_file = os.path.join(current_dir, "assets/ligand_3cyx.pdb")
    
    card_protein = DataCard(address='', file_type='pdb', data_type='text/plain')
    card_ligand = DataCard(address='', file_type='pdb', data_type='text/plain')
    
    pdb_address = disk_datastore.upload_data("protein_test.pdb", pdb_test_file, card_protein)
    ligand_address = disk_datastore.upload_data("ligand_test.pdb", ligand_test_file, card_ligand)
    
    # Test with different exhaustiveness values
    for exhaustiveness in [1, 5, 10]:
        result_address = generate_pose(
            protein_address=pdb_address,
            ligand_address=ligand_address,
            output=f'test_output_exh_{exhaustiveness}',
            exhaustiveness=exhaustiveness,
            num_modes=1
        )
        
        # Verify result
        assert result_address.startswith('deepchem://')
        
        # Download and verify results
        results_data = disk_datastore.get(result_address)
        results = json.loads(results_data)
        
        assert results['docking_method'] == 'VINA'
        assert results['num_modes'] >= 0


def test_generate_pose_nested_full_address(disk_datastore):
    """Test VINA pose generation with nested full addresses."""
    config.set_datastore(disk_datastore)
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    pdb_test_file = os.path.join(current_dir, "assets/cleaned_3cyx.pdb")
    ligand_test_file = os.path.join(current_dir, "assets/ligand_3cyx.pdb")
    
    card_protein = DataCard(address='', file_type='pdb', data_type='text/plain')
    card_ligand = DataCard(address='', file_type='pdb', data_type='text/plain')
    
    # Upload with nested paths
    pdb_address = disk_datastore.upload_data("test docking/protein_test.pdb", pdb_test_file, card_protein)
    ligand_address = disk_datastore.upload_data("test docking/ligand_test.pdb", ligand_test_file, card_ligand)
    
    assert pdb_address == "deepchem://test/user/test docking/protein_test.pdb"
    assert ligand_address == "deepchem://test/user/test docking/ligand_test.pdb"
    
    # Test pose generation
    result_address = generate_pose(
        protein_address=pdb_address,
        ligand_address=ligand_address,
        output="deepchem://test/user/test docking/docking_results",
        exhaustiveness=1,
        num_modes=1
    )
    
    assert result_address.startswith('deepchem://')
    assert result_address.endswith('_results.json')
    
    # Download and verify results
    results_data = disk_datastore.get(result_address)
    results = json.loads(results_data)
    
    assert results['docking_method'] == 'VINA'
    assert results['num_modes'] >= 0


def test_generate_pose_score_formatting(disk_datastore):
    """Test that VINA scores are formatted correctly"""
    config.set_datastore(disk_datastore)
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    pdb_test_file = os.path.join(current_dir, "assets/cleaned_3cyx.pdb")
    ligand_test_file = os.path.join(current_dir, "assets/ligand_3cyx.pdb")
    
    card_protein = DataCard(address='', file_type='pdb', data_type='text/plain')
    card_ligand = DataCard(address='', file_type='pdb', data_type='text/plain')
    
    pdb_address = disk_datastore.upload_data("protein_test.pdb", pdb_test_file, card_protein)
    ligand_address = disk_datastore.upload_data("ligand_test.pdb", ligand_test_file, card_ligand)
    
    # Test with multiple modes to verify score formatting
    result_address = generate_pose(
        protein_address=pdb_address,
        ligand_address=ligand_address,
        output='test_score_formatting',
        exhaustiveness=1,
        num_modes=2
    )
    
    # Download and verify results
    results_data = disk_datastore.get(result_address)
    results = json.loads(results_data)
    
    # Check exact score format: 'mode %s' % (i + 1)
    scores = results['scores']
    
    # Verify mode keys follow the exact format
    expected_mode_keys = ['mode 1', 'mode 2']
    for expected_key in expected_mode_keys:
        assert expected_key in scores
    
    # Verify score structure for each mode
    for mode_key in scores:
        assert 'affinity (kcal/mol)' in scores[mode_key]
        # Verify score is numeric and reasonable (VINA scores are typically negative)
        score_value = scores[mode_key]['affinity (kcal/mol)']
        assert isinstance(score_value, (int, float))
        # VINA scores are typically negative (better binding = more negative)
        # But we don't enforce this strictly as it depends on the specific system


def test_generate_pose_error_handling(disk_datastore):
    """Test error handling in VINA pose generation."""
    config.set_datastore(disk_datastore)
    
    # Test with invalid addresses
    with pytest.raises(ValueError) as context:
        generate_pose(
            protein_address="",
            ligand_address="deepchem://test/ligand.pdb",
            output='test_error'
        )
    assert "Protein and/or ligand input is required" in str(context.exception)
    
    with pytest.raises(ValueError) as context:
        generate_pose(
            protein_address="deepchem://test/protein.pdb",
            ligand_address="",
            output='test_error'
        )
    assert "Protein and/or ligand input is required" in str(context.exception)
    
    with pytest.raises(ValueError) as context:
        generate_pose(
            protein_address=None,
            ligand_address="deepchem://test/ligand.pdb",
            output='test_error'
        )
    assert "Protein and/or ligand input is required" in str(context.exception)


def test_generate_pose_progress_logging(disk_datastore):
    """Test that progress logging is properly integrated."""
    config.set_datastore(disk_datastore)
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    pdb_test_file = os.path.join(current_dir, "assets/cleaned_3cyx.pdb")
    ligand_test_file = os.path.join(current_dir, "assets/ligand_3cyx.pdb")
    
    card_protein = DataCard(address='', file_type='pdb', data_type='text/plain')
    card_ligand = DataCard(address='', file_type='pdb', data_type='text/plain')
    
    pdb_address = disk_datastore.upload_data("protein_test.pdb", pdb_test_file, card_protein)
    ligand_address = disk_datastore.upload_data("ligand_test.pdb", ligand_test_file, card_ligand)
    
    # Test that progress logging doesn't interfere with functionality
    result_address = generate_pose(
        protein_address=pdb_address,
        ligand_address=ligand_address,
        output='test_progress_logging',
        exhaustiveness=1,
        num_modes=1
    )
    
    # Verify the function still works despite progress logging
    assert result_address.startswith('deepchem://')
    
    # Download and verify results
    results_data = disk_datastore.get(result_address)
    results = json.loads(results_data)
    
    assert results['docking_method'] == 'VINA'
    assert results['num_modes'] >= 0


def test_generate_pose_data_card_integration(disk_datastore):
    """Test that DataCard integration works correctly for docking results."""
    config.set_datastore(disk_datastore)
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    pdb_test_file = os.path.join(current_dir, "assets/cleaned_3cyx.pdb")
    ligand_test_file = os.path.join(current_dir, "assets/ligand_3cyx.pdb")
    
    card_protein = DataCard(address='', file_type='pdb', data_type='text/plain')
    card_ligand = DataCard(address='', file_type='pdb', data_type='text/plain')
    
    pdb_address = disk_datastore.upload_data("protein_test.pdb", pdb_test_file, card_protein)
    ligand_address = disk_datastore.upload_data("ligand_test.pdb", ligand_test_file, card_ligand)
    
    # Test pose generation
    result_address = generate_pose(
        protein_address=pdb_address,
        ligand_address=ligand_address,
        output='test_datacard',
        exhaustiveness=1,
        num_modes=1
    )
    
    # Verify result
    assert result_address.startswith('deepchem://')
    
    # Check that DataCard was created for the results
    result_card_address = result_address + '.cdc'
    result_card = disk_datastore.get(result_card_address)
    
    # Verify DataCard properties
    assert result_card.file_type == 'json'
    assert result_card.data_type == 'docking results'

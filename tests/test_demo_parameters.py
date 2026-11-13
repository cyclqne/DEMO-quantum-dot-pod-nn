"""Public demo controls preserve research defaults and support fixed expert counts."""
from pathlib import Path
import numpy as np
import pytest
import yaml
from qdpod.config import load_config
from qdpod.clustering import choose_validated_state_clusters, expert_width


def test_annotated_demo_file_preserves_last_research_defaults():
    annotated=load_config('configs/demo_parameters.yaml')
    original=load_config('configs/default_40state_tanh.yaml')
    assert annotated == original
    assert (annotated['physics']['grid'], annotated['physics']['states'],
            annotated['pod']['modes']) == (49,40,80)
    assert (annotated['network']['fast_epochs'],
            annotated['network']['finetune_epochs']) == (850,200)
    assert annotated['network']['activation'] == 'tanh'


def test_can_set_exact_number_of_local_networks():
    rng=np.random.default_rng(233)
    train=rng.random((40,80))
    val=train+rng.normal(0,.001,size=train.shape)
    config=load_config('configs/demo_parameters.yaml')
    config['clustering']['fixed_k']=4
    labels,rows=choose_validated_state_clusters(train,val,config)
    assert len(np.unique(labels)) == 4
    assert len(rows) == 1 and rows[0]['k'] == 4 and rows[0]['selected']


def test_fixed_k_validation(tmp_path):
    config=load_config('configs/demo_parameters.yaml')
    for invalid in [0,1,41,True,'four']:
        config['clustering']['fixed_k']=invalid
        f=tmp_path/'custom.yaml'
        f.write_text(yaml.safe_dump(config))
        with pytest.raises(ValueError):load_config(f)


def test_expert_width_cutoffs_are_tunable(tmp_path):
    config=load_config('configs/demo_parameters.yaml')
    config['network']['expert_width_state_thresholds'] = {
        'small_max':5,'large_min':8,'very_large_min':11}
    assert expert_width(5,config)==96
    assert expert_width(6,config)==128
    assert expert_width(8,config)==160
    assert expert_width(11,config)==192
    config['network']['expert_width_state_thresholds']['large_min']=4
    f=tmp_path/'bad.yaml'
    f.write_text(yaml.safe_dump(config))
    with pytest.raises(ValueError):load_config(f)


def test_config_documents_all_default_active_leaf_parameters():
    text=Path('configs/demo_parameters.yaml').read_text()
    # The annotated parameter file should remain approachable and self-contained.
    assert '# fixed_k: null' in text
    assert 'Randomized-SVD power iterations' in text
    assert 'Signed 3D wavefunction' not in text or '3D wavefunction' in text
    assert 'network:' in text and 'runtime:' in text

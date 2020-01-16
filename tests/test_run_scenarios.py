import json
import pytest
from ulmfit_attention import scenarios


@pytest.mark.slow
def test_run_from_jsons():
    task = json.load(open('./tasks/imdb_1k_sample_single.json', 'r'))
    conf = json.load(open('./configs/imdb_1k.json', 'r'))
    all_params = dict(**conf, **task)
    scenario = scenarios.Scenario.from_config(task['scenario'])
    res = scenario.single_run(all_params)
    assert res[0] > 0.9

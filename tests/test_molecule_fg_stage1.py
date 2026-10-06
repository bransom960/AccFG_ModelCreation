import json
import sys
from pathlib import Path

STAGE1_DIR = Path(__file__).resolve().parents[1] / 'molecule-fg-data' / 'stage1'
if str(STAGE1_DIR) not in sys.path:
    sys.path.insert(0, str(STAGE1_DIR))

import extract_smiles  # noqa: E402


def test_extract_writes_cid_and_smiles(tmp_path):
    src = tmp_path / 'part.jsonl'
    src.write_text('\n'.join(json.dumps(r) for r in [
        {'cid': '176000001', 'smiles': 'CCO', 'c-smiles': 'CCO'},
        {'cid': 176000002, 'smiles': 'C/C=C\\C'},   # bare-number cid; escaped backslash
        {'smiles': 'CCN'},                          # no cid: skipped and counted
    ]) + '\n')
    out = tmp_path / 'part.smi'

    stats = extract_smiles.extract_one((str(src), str(out), 'smiles', 'cid', False, 1.0, 0))

    assert out.read_text().splitlines() == ['176000001\tCCO', '176000002\tC/C=C\\C']
    assert stats['written'] == 2
    assert stats['no_id'] == 1


def _write_jsonl(path, records):
    path.write_text('\n'.join(json.dumps(r) for r in records) + '\n')


def test_salts_are_not_eligible():
    pat, id_pat = extract_smiles._pattern('smiles'), extract_smiles._id_pattern('cid')
    line = json.dumps({'cid': '1', 'smiles': 'CC(=O)[O-].[Na+]'}).encode()
    assert extract_smiles.parse_record(line, pat, id_pat)[0] == 'salt'


def test_quotas_sum_to_target_and_fit_each_file():
    quotas = extract_smiles.allocate([10, 0, 5, 40], 30, seed=0)
    assert sum(quotas) == 30
    assert all(0 <= q <= n for q, n in zip(quotas, [10, 0, 5, 40]))
    assert quotas == extract_smiles.allocate([10, 0, 5, 40], 30, seed=0)  # seeded


def test_target_sample_is_exact_and_skips_salts(tmp_path):
    src = tmp_path / 'part.jsonl'
    records = [{'cid': str(i), 'smiles': 'C' * (i % 5 + 1)} for i in range(50)]
    records += [{'cid': str(100 + i), 'smiles': 'CCO.Cl'} for i in range(20)]  # salts
    _write_jsonl(src, records)

    counts = extract_smiles.count_one((str(src), 'smiles', 'cid'))
    assert (counts['ok'], counts['salt']) == (50, 20)

    out = tmp_path / 'part.smi'
    r = extract_smiles.sample_one((str(src), str(out), 'smiles', 'cid', 12, 50, 0, False))

    lines = out.read_text().splitlines()
    assert r['written'] == 12 and len(lines) == 12
    assert len({line.split('\t')[0] for line in lines}) == 12  # distinct molecules
    assert not any('.' in line for line in lines)


def test_unparseable_picks_are_replaced(tmp_path):
    src = tmp_path / 'part.jsonl'
    # every other SMILES has an unclosed ring, which RDKit cannot parse
    _write_jsonl(src, [{'cid': str(i), 'smiles': 'C1CC' if i % 2 else 'CCO'} for i in range(40)])
    out = tmp_path / 'part.smi'

    r = extract_smiles.sample_one((str(src), str(out), 'smiles', 'cid', 15, 40, 0, True))

    lines = out.read_text().splitlines()
    assert r['written'] == 15 and len(lines) == 15
    assert all(line.endswith('\tCCO') for line in lines)
    assert r['unparseable'] > 0

import json
from tqdm import tqdm
from pathlib import Path

class IO:
    @staticmethod
    def is_valid_file(filepath):
        filepath = Path(filepath)
        return filepath.exists() and filepath.stat().st_size > 0

    def load(path):
        raise NotImplementedError

    def dump(data, path):
        raise NotImplementedError

class JsonLine(IO):
    @staticmethod
    def load(path, use_tqdm=False):
        with open(path) as rf:
            lines = rf.read().splitlines()
        if use_tqdm:
            lines = tqdm(lines, ncols=100, desc='Load JsonLine')
        return [json.loads(l) for l in lines]

    @staticmethod
    def dump(instances, path):
        assert type(instances) == list
        lines = [json.dumps(d, ensure_ascii=False) for d in instances]
        with open(path, 'w') as wf:
            wf.write('\n'.join(lines))

import json


def load_json(path: str):
    with open(path) as f:
        content = f.read()
    data = json.loads(content)
    return data


def save_json(data, path: str):
    dump = json.dumps(data)
    with open(path, "w") as f:
        f.write(dump)

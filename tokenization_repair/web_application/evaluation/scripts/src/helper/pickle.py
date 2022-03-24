import pickle


def dump_object(obj, path):
    with open(path, "wb") as file:
        pickle.dump(obj, file)


def load_object(path):
    with open(path, "rb") as file:
        return pickle.load(file)

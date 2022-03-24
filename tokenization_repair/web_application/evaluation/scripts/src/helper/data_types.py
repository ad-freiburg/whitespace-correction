
def is_float(string: str) -> bool:
    try:
        float(string)
        return True
    except ValueError:
        return False

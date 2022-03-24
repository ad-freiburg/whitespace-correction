import time


def timestamp() -> float:
    return time.time()


def time_diff(start_time: float) -> float:
    return timestamp() - start_time

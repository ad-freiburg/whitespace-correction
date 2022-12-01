from typing import Union


def constrain(a: Union[int, float], minimum: Union[int, float], maximum: Union[int, float]) -> Union[int, float]:
    assert minimum <= maximum
    return min(max(a, minimum), maximum)

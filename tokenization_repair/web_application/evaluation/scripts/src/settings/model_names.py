FWD = "conll.fwd1024"
FWD_ROBUST = "conll.fwd1024.ocr+spelling"
BWD = "bwd1024"
BWD_ROBUST = "bwd1024_noise0.2"
BIDIR = "conll.labeling"
BIDIR_ROBUST = "conll.labeling.ocr+spelling"


def unidirectional_model_name(backward: bool, robust: bool) -> str:
    if backward:
        if robust:
            return BWD_ROBUST
        else:
            return BWD
    else:
        if robust:
            return FWD_ROBUST
        else:
            return FWD


def bidirectional_model_name(robust: bool) -> str:
    return BIDIR_ROBUST if robust else BIDIR

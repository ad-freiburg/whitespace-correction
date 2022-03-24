from src.settings import paths


def get_settings():
    settings = {"paths": {"fwd": paths.MODEL_FOLDER + "wiki_fwd_noDropout",
                          "bwd": paths.MODEL_FOLDER + "wiki_bwd_noDropout",
                          "bidirectional": paths.MODEL_FOLDER + "wiki_bidir",
                          "bidirectional_sigmoid": paths.MODEL_FOLDER + "wiki_bidir_sigmoid"},
                "thresholds": {"tokenization": {"mixed": {"insertion": 0.833474,
                                                          "deletion": 0.383286},
                                                "bidirectional_sigmoid": {"insertion": 0.837327,
                                                                          "deletion": 0.994371},
                                                "combined": {"insertion": 0.999962,
                                                             "deletion": 0.374470}
                                                },
                               "spelling": {"mixed": {"insertion": 0.878156,
                                                      "deletion": 0.997940},
                                            "bidirectional_sigmoid": {"insertion": 0.878156,
                                                                      "deletion": 0.999977},
                                            "combined": {"insertion": 0.999314,
                                                         "deletion": 0.997761}
                                            }
                               }
                }
    return settings

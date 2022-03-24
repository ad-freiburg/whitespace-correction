from src.sequence.corruption import CorruptionType
from src.evaluation.metrics import precision_recall_f1, ed_fraction_resolved


def print_sequence_result(result):
    print("(sequence) (insertions) %i TP, %i FP, %i FN" %
          (result.num_tp(CorruptionType.INSERTION),
           result.num_fp(CorruptionType.INSERTION),
           result.num_fn(CorruptionType.INSERTION)))
    print("(sequence) (deletions)  %i TP, %i FP, %i FN" %
          (result.num_tp(CorruptionType.DELETION),
           result.num_fp(CorruptionType.DELETION),
           result.num_fn(CorruptionType.DELETION)))
    print("(sequence) (total)      %i TP, %i FP, %i FN" %
          (result.num_tp(),
           result.num_fp(),
           result.num_fn()))
    print("(sequence) Edit distance changed from %i to %i." % (result.ed_before, result.ed_after))
    print("(sequence) Edit distance fraction resolved = %.4f" % ed_fraction_resolved(result.ed_before, result.ed_after))
    print("(sequence) Runtime = %.2f sec" % result.runtime)


def evaluator_string(evaluator):
    string = ""
    # insertions f1
    tp_insertions, fp_insertions, fn_insertions = evaluator.nums_tp_fp_fn(type=CorruptionType.INSERTION)
    precision_insertions, recall_insertions, f1_insertions = \
        precision_recall_f1(tp_insertions, fp_insertions, fn_insertions)
    string += ("(total)    (insertions) %i TP, %i FP, %i FN" % (tp_insertions, fp_insertions, fn_insertions)) + "\n"
    string += ("(total)    (insertions) precision = %.4f" % precision_insertions) + "\n"
    string += ("(total)    (insertions) recall = %.4f" % recall_insertions) + "\n"
    string += ("(total)    (insertions) f1 = %.4f" % f1_insertions) + "\n"
    # deletions f1
    tp_deletions, fp_deletions, fn_deletions = evaluator.nums_tp_fp_fn(type=CorruptionType.DELETION)
    precision_deletions, recall_deletions, f1_deletions = \
        precision_recall_f1(tp_deletions, fp_deletions, fn_deletions)
    string += ("(total)    (deletions)  %i TP, %i FP, %i FN" % (tp_deletions, fp_deletions, fn_deletions)) + "\n"
    string += ("(total)    (deletions)  precision = %.4f" % precision_deletions) + "\n"
    string += ("(total)    (deletions)  recall = %.4f" % recall_deletions) + "\n"
    string += ("(total)    (deletions)  f1 = %.4f" % f1_deletions) + "\n"
    # total f1
    tp_total, fp_total, fn_total = evaluator.nums_tp_fp_fn()
    precision_total, recall_total, f1_total = \
        precision_recall_f1(tp_total, fp_total, fn_total)
    string += ("(total)    (total)      %i TP, %i FP, %i FN" % (tp_total, fp_total, fn_total)) + "\n"
    string += ("(total)    (total)      precision = %.4f" % precision_total) + "\n"
    string += ("(total)    (total)      recall = %.4f" % recall_total) + "\n"
    string += ("(total)    (total)      f1 = %.4f" % f1_total) + "\n"
    # edit distance
    #string += ("(total)    Mean edit distance before = %.2f" % evaluator.mean_ed_before()) + "\n"
    #string += ("(total)    Mean edit distance after  = %.2f" % evaluator.mean_ed_after()) + "\n"
    #string += ("(total)    Edit distance fraction resolved = %.4f" %
    #      ed_fraction_resolved(evaluator.mean_ed_before(), evaluator.mean_ed_after())) + "\n"
    # sequence accuracy
    string += ("(total)    Sequence accuracy = %.4f (%i/%i)" %
          (evaluator.sequence_accuracy(), evaluator.num_correct_sequences(), evaluator.num_sequences())) + "\n"
    # runtime
    #string += ("(total)    Mean runtime per sequence = %.2f sec" % evaluator.mean_runtime())
    return string, f1_total, evaluator.sequence_accuracy()


def print_evaluator(evaluator):
    output_str, f1, acc = evaluator_string(evaluator)
    print(output_str)
    return f1, acc

$(document).ready(function() {
    console.log("document ready");

    // GET BENCHMARKS
    benchmarks = [];

    $.get("./org_benchmarks/", function(data) {
        $(data).find("a").each(function() {
            name = $(this).attr("href");
            if (!name.endsWith(".txt") && name[0] !== "?" && name[0] !== "/") {
                name = name.substring(0, name.length - 1);
                benchmarks.push(name);
                console.log(name);
                $("#select_benchmark").append(new Option(name, name));
            }
        });
        show_overview_table("development");
        show_overview_table("test");
        set_prediction_options();
        show_benchmark_results_table();
    });

    // GET PREDICTION FILES

    $("#select_benchmark").change(function() {
        set_prediction_options();
        show_benchmark_results_table();
    });

    $("#select_subset").change(function() {
        set_prediction_options();
        show_benchmark_results_table();
    });

    $("#select_predictions").change(function() {
        create_table();
    });

    $("#ignore_punctuation").change(function() {
        create_table();
    });

    $("#hide_zeros").change(function () {
        hide_zero_rows();
    });

    APPROACH_ORDER = [
        "do_nothing.txt",
        "wordsegment.txt",
        "google.txt",
        "bid+.txt",
        "the-one.txt",
        "eo_large_arxiv_with_errors.txt"
    ];
    BENCHMARK_ORDER = [
        "ACL",
        "arXiv.OCR",
        "arXiv.pdftotext",
        "Wiki",
        "Wiki.typos",
        "Wiki.typos.no_spaces",
        "doval",
    ];
});

function set_prediction_options() {
    benchmark = $("#select_benchmark option:selected").val();
    console.log(benchmark);

    subset = $("#select_subset option:selected").val();
    console.log(subset);

    prediction_files = [];
    $("#select_predictions").empty();

    read_benchmark();

    results_dir = "./org_results/" + benchmark + "/" + subset + "/";
    $.get(results_dir, function(data) {
        $(data).find("a").each(function() {
            name = $(this).attr("href");
            if (name.endsWith(".txt")) {
                prediction_files.push(name);
            }
        });
        for (approach of APPROACH_ORDER) {
            if (prediction_files.includes(approach)) {
                $("#select_predictions").append(new Option(approach, approach));
            }
        }
        for (file of prediction_files) {
            if (!APPROACH_ORDER.includes(file)) {
                $("#select_predictions").append(new Option(file, file));
            }
        }
        $("#select_predictions").prop("selectedIndex", -1);
        $("#table").html("select a file above");
    });
}

function read_benchmark() {
    benchmark_dir = "./org_benchmarks/" + benchmark + "/" + subset + "/";
    $.get(benchmark_dir + "correct.txt", function(data) {
        correct_sequences = []
        for (sequence of data.split("\n")) {
            if (sequence.length > 0) {
                correct_sequences.push(sequence);
            }
        }
    });
    $.get(benchmark_dir + "corrupt.txt", function(data) {
        corrupt_sequences = data.split("\n");
    });
}

function get_inserted_nonspace_positions(original, misspelled) {
    original_tokens = original.split(" ");
    misspelled_tokens = misspelled.split(" ");
    pos = 0;
    differences = new Set();
    for (var i = 0; i < original_tokens.length; i++) {
        if (original_tokens[i].length < misspelled_tokens[i].length) {
            if (misspelled_tokens[i].substring(1) === original_tokens[i]) {
                differences.add(pos);
            } else {
                for (var j = 0; j < misspelled_tokens[i].length; j++) {
                    if (j === original_tokens[i].length || misspelled_tokens[i][j] !== original_tokens[i][j]) {
                        differences.add(pos + j);
                        break;
                    }
                }
            }
        }
        pos += misspelled_tokens[i].length;
    }
    return differences;
}

function remove_nonspace_positions(sequence, positions) {
    txt = ""
    pos = 0;
    for (var i = 0; i < sequence.length; i++) {
        if (sequence[i] === " ") {
            if (!txt.endsWith(" ")) {
                txt += sequence[i];
            }
        } else if (positions.has(pos)) {
            pos += 1;
        } else {
            pos += 1;
            txt += sequence[i];
        }
    }
    return txt;
}

function create_table() {
    $("#table").html("evaluating...");
    $("#sequences").html("");

    error_tolerant = benchmark.startsWith("0.1") || benchmark.startsWith("WikiT") || benchmark.startsWith("Wiki_typos");
    console.log("error tolerant: " + error_tolerant);
    if (error_tolerant) {
        if (subset === "development") {
            original_sequences = wiki_development_sequences;
        } else {
            original_sequences = wiki_test_sequences;
        }
    }

    selected = $("#select_predictions option:selected").val();
    predictions_file = results_dir + selected;
    console.log(predictions_file);

    $.get(predictions_file, function(data) {
        predicted_sequences = data.split("\n");

        n = Math.min(correct_sequences.length, corrupt_sequences.length, predicted_sequences.length);
        console.log(n);

        n_corrupt = 0;
        n_correct = 0;

        n_sequences = 0;
        n_tp = 0;
        n_fp = 0;
        n_fn = 0;

        table = "<table>\n";
        table += "<th>ID</th>"
        table += "<th>INPUT</th>"
        table += "<th>GROUND TRUTH</th>"
        table += "<th>PREDICTED</th>"
        table += "<th>TP</th>"
        table += "<th>FP</th>"
        table += "<th>FN</th>"
        table += "<th>CORR</th>\n"

        for (var i = 0; i < n; i++) {
            if (corrupt_sequences[i].replaceAll(' ', '') === predicted_sequences[i].replaceAll(' ', '')) {
                n_sequences += 1;

                // tolerance preprocessing

                if (error_tolerant) {
                    original_sequence = original_sequences[i];
                    diff_positions =  get_inserted_nonspace_positions(original_sequences[i], correct_sequences[i]);

                    correct_preprocessed = remove_nonspace_positions(correct_sequences[i], diff_positions);
                    corrupt_preprocessed = remove_nonspace_positions(corrupt_sequences[i], diff_positions);
                    predicted_preprocessed = remove_nonspace_positions(predicted_sequences[i], diff_positions);
                } else {
                    original_sequence = correct_sequences[i];
                    correct_preprocessed = correct_sequences[i];
                    corrupt_preprocessed = corrupt_sequences[i];
                    predicted_preprocessed = predicted_sequences[i];
                }

                // ground truth and prediction

                [ground_truth, highlight_true] = get_differences(corrupt_preprocessed, correct_preprocessed);
                [predictions, highlight_predicted] = get_differences(corrupt_preprocessed, predicted_preprocessed);
                [_unused, wrong_positions] = get_differences(correct_preprocessed, predicted_preprocessed);
                predicted_highlighted = highlight_positions_with_truth(predicted_sequences[i], highlight_predicted, wrong_positions, predicted_preprocessed);

                // evaluate TP, FP, FN

                tp = ground_truth.filter(x => predictions.includes(x));
                fp = predictions.filter(x => !ground_truth.includes(x));
                fn = ground_truth.filter(x => !predictions.includes(x));

                is_corrupted = tp.length + fn.length > 0;
                is_correct = fp.length + fn.length === 0;

                n_tp += tp.length;
                n_fp += fp.length;
                n_fn += fn.length;

                if (is_corrupted) {
                    n_corrupt += 1;
                }
                if (is_correct) {
                    n_correct += 1;
                }

                // row class

                row_class = null;
                if (tp.length == 0 && fp.length == 0 && fn.length == 0) {
                    row_class = "all_zeros";
                }

                // colors

                corrupt_color = "black";
                predicted_color = "black";
                if (is_corrupted) {
                    corrupt_color = "red";
                    if (is_correct) {
                        predicted_color = "green";
                    }
                }
                if (!is_correct) {
                    predicted_color = "red";
                }

                // sequence result

                if (is_correct) {
                    sequence_result = "yes";
                } else {
                    sequence_result = "no";
                }

                // row
                if (row_class == null) {
                    row = "<tr>";
                } else {
                    row = "<tr class=\"" + row_class + "\">";
                }

                // .. sequences
                row += "<td>" + (i + 1) + "</td>";
                row += "<td style=\"color:" + corrupt_color + "\">" + corrupt_sequences[i] + "</td>";
                row += "<td>" + original_sequence + "</td>";
                row += "<td>" + predicted_highlighted + "</td>";
                // .. evaluation counts
                row += "<td>" + tp.length + "</td>";
                row += "<td>" + fp.length + "</td>";
                row += "<td>" + fn.length + "</td>";
                // .. sequence result
                row += "<td style=\"color:" + predicted_color + "\">" + sequence_result + "</td>";
                // .. end row
                row += "</tr>";
                table += row + "\n";
            }
        }
        table += "</table>";

        precision = n_tp / (n_tp + n_fp);
        recall = n_tp / (n_tp + n_fn);
        f1 = 2 * precision * recall / (precision + recall);

        evaluation = "<table>";
        evaluation = "<tr><td>corrupt sequences</td><td>" + (n_corrupt / n_sequences).toFixed(4) + " (" + n_corrupt + "/" + n_sequences + ")</td></tr>\n";
        evaluation += "<tr><td>sequence accuracy</td><td>" + (n_correct / n_sequences).toFixed(4) + " (" + n_correct + "/" + n_sequences + ")</td></tr>\n";
        evaluation += "<tr><td>true positives</td><td>" + n_tp + "</td></tr>\n";
        evaluation += "<tr><td>false positives</td><td>" + n_fp + "</td></tr>\n";
        evaluation += "<tr><td>false negatives</td><td>" + n_fn + "</td></tr>\n";
        evaluation += "<tr><td>precision</td><td>" + precision.toFixed(4) + "</td></tr>\n";
        evaluation += "<tr><td>recall</td><td>" + recall.toFixed(4) + "</td></tr>\n";
        evaluation += "<tr><td>F1-score</td><td>" + f1.toFixed(4) + "</td></tr>\n";
        evaluation += "</table>";

        $("#table").html(evaluation);
        $("#sequences").html(table);

        hide_zero_rows();
    });
}

function isalnum(char) {
    return char.match(/^[0-9a-z]+$/);
}

function get_differences(a, b) {
    ignore_punctuation = $("#ignore_punctuation").is(":checked");
    diff_positions_a = [];
    diff_positions_b = [];
    var i = 0;
    var j = 0;
    while (i < a.length && j < b.length) {
        if (a[i] == b[j]) {
            i += 1;
            j += 1;
        } else {
            do_ignore = false;
            if (ignore_punctuation) {
                if (a[i] == " ") {
                    if (!isalnum(a[i-1]) || !isalnum(a[i+1])) {
                        do_ignore = true;
                    }
                } else {
                    if (!isalnum(b[j-1]) || !isalnum(b[j+1])) {
                        do_ignore = true;
                    }
                }
            }
            if (!do_ignore) {
                diff_positions_a.push(i);
                diff_positions_b.push(j);
            }
            if (a[i] == " ") {
                i += 1;
            } else {
                j += 1;
            }
        }
    }
    return [diff_positions_a, diff_positions_b];
}

function highlight_positions(text, positions) {
    for (pos of positions.reverse()) {
        text = text.substring(0, pos) + "<u>" + text[pos] + "</u>" + text.substring(pos + 1);
    }
    return text;
}

function union(set1, set2) {
    _union = new Set(set1);
    for (elem of set2) {
        _union.add(elem);
    }
    return _union;
}

function compare_numbers(a, b) {
    return a - b;
}

function highlight_positions_with_truth(text, predicted_positions, wrong_positions, preprocessed_sequence) {
    predicted_positions = new Set(predicted_positions);
    wrong_positions = new Set(wrong_positions);
    html = "";
    pos_text = 0;
    pos_preprocessed = 0;
    while (pos_text < text.length) {
        if (pos_preprocessed == preprocessed_sequence.length || text[pos_text] != preprocessed_sequence[pos_preprocessed]) {
            html += "<strike>" + text[pos_text] + "</strike>";
            pos_text += 1;
        } else {
            if (wrong_positions.has(pos_preprocessed)) {
                color = "red";
            } else if (predicted_positions.has(pos_preprocessed)) {
                color = "green";
            } else {
                color = null;
            }
            if (predicted_positions.has(pos_preprocessed)) {
                html_element = "u";
            } else {
                html_element = "span";
            }
            if (color == null) {
                html += text[pos_text];
            } else {
                open_tag = "<" + html_element + " style=\"background-color:" + color + "\">";
                close_tag = "</" + html_element + ">";
                html += open_tag + text[pos_text] + close_tag;
            }
            pos_text += 1;
            pos_preprocessed += 1;
        }
    }
    return html;
}

function hide_zero_rows() {
    if ($("#hide_zeros").is(":checked")) {
        $(".all_zeros").hide();
    } else {
        $(".all_zeros").show();
    }
}

function percent(fraction) {
    return (fraction * 100).toFixed(2) + " %";
}

function show_benchmark_results_table() {
    benchmark = $("#select_benchmark").val();
    subset = $("#select_subset").val();
    results_path = "./org_results/" + benchmark + "/" + subset + "/results.json";
    console.log(results_path);
    table_body = "";
    $.getJSON(results_path, function(results) {
        for (approach of show_approaches) {
            if (approach in results) {
                key = approach;
                console.log(results[key]);
                table_body += "<tr>";
                table_body += "<td>" + key + "</td>";
                table_body += "<td>" + percent(results[key].precision) + "</td>";
                table_body += "<td>" + percent(results[key].recall) + "</td>";
                table_body += "<td>" + percent(results[key].f1) + "</td>";
                table_body += "<td>" + percent(results[key].sequence_accuracy) + "</td>";
                table_body += "</tr>";
            }
        }
        $("#tbody_benchmark_results").html(table_body);
    });
}

function show_overview_table(subset) {
    benchmark_results = {};
    for (benchmark of benchmarks) {
        results_path = "./org_results/" + benchmark + "/" + subset + "/results.json";
        console.log(results_path);
        $.ajax({url: results_path,
                async: false,
                success: function(result) {
                    benchmark_results[benchmark] = result;
                }
        });
    }
    console.log(benchmark_results);
    overview_benchmarks = Object.keys(benchmark_results);
    overview_benchmarks.sort();
    overview_benchmarks.reverse();
    show_benchmarks = [];
    for (benchmark of BENCHMARK_ORDER) {
        if (overview_benchmarks.includes(benchmark)) {
            show_benchmarks.push(benchmark);
        }
    }
    for (benchmark of overview_benchmarks) {
        if (!BENCHMARK_ORDER.includes(benchmark)) {
            show_benchmarks.push(benchmark);
        }
    }
    overview_approaches = new Set();
    for (benchmark of overview_benchmarks) {
        for (approach of Object.keys(benchmark_results[benchmark])) {
            overview_approaches.add(approach);
        }
    }
    overview_approaches = Array.from(overview_approaches);
    overview_approaches.sort();
    overview_approaches.reverse();
    show_approaches = [];
    for (approach of APPROACH_ORDER) {
        if (overview_approaches.includes(approach)) {
            show_approaches.push(approach);
        }
    }
    for (approach of overview_approaches) {
        if (!APPROACH_ORDER.includes(approach)) {
            show_approaches.push(approach);
        }
    }
    // table head
    thead = "<tr>";
    thead += "<th rowspan=\"2\">Approach</th>";
    for (benchmark of show_benchmarks) {
        thead += "<th colspan=\"2\">" + benchmark + "</th>";
    }
    thead += "</tr><tr>";
    for (benchmark of show_benchmarks) {
        thead += "<th>F1</th>";
        thead += "<th>Acc</th>";
    }
    thead += "</tr>";
    $("#thead_overview_table_" + subset).html(thead);
    // table body
    tbody = "";
    for (approach of show_approaches) {
        row = "<tr>";
        row += "<td>" + approach + "</td>";
        for (benchmark of show_benchmarks) {
            if (benchmark_results[benchmark][approach]) {
                f1 = percent(benchmark_results[benchmark][approach].f1);
                acc = percent(benchmark_results[benchmark][approach].sequence_accuracy);
            } else {
                f1 = "-";
                acc = "-";
            }
            row += "<td>" + f1 + "</td>";
            row += "<td>" + acc + "</td>";
        }
        row += "</tr>";
        tbody += row;
    }
    $("#tbody_overview_table_" + subset).html(tbody);
}

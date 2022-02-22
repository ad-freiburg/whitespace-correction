| Benchmark            | Split   |   Runtime in seconds |   Number of samples |   File size in KB |
|:---------------------|:--------|---------------------:|--------------------:|------------------:|
| ACL                  | test    |              3.64202 |                 462 |           56.0674 |
| arXiv.OCR            | test    |             58.2333  |                9360 |          882.581  |
| arXiv.pdftotext      | test    |             58.9351  |                9371 |          878.843  |
| doval                | test    |              5.68979 |                1000 |           81.6943 |
| Wiki                 | test    |             63.6472  |                9997 |          914.075  |
| Wiki.typos.no_spaces | test    |             58.818   |                9997 |          775.755  |
| Wiki.typos           | test    |             65.7879  |                9997 |          913.895  |

| Model                                                    |   Total runtime in seconds |   samples/s |   s/KB |
|:---------------------------------------------------------|---------------------------:|------------:|-------:|
| eo_large_arxiv_no_errors_finetuned_temp_thresh_optimized |                    314.753 |     159.439 | 0.0699 |
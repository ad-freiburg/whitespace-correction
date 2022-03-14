| Benchmark            | Split   |   Runtime in seconds |   Number of samples |   File size in KB |
|:---------------------|:--------|---------------------:|--------------------:|------------------:|
| ACL                  | test    |              1.18009 |                 462 |           56.0674 |
| arXiv.OCR            | test    |             19.1794  |                9360 |          882.581  |
| arXiv.pdftotext      | test    |             19.1762  |                9371 |          878.843  |
| doval                | test    |              1.84769 |                1000 |           81.6943 |
| Wiki                 | test    |             19.1656  |                9997 |          914.075  |
| Wiki.typos.no_spaces | test    |             16.9355  |                9997 |          775.755  |
| Wiki.typos           | test    |             19.5895  |                9997 |          913.895  |

| Model                              |   Total runtime in seconds |   samples/s |     s/KB |
|:-----------------------------------|---------------------------:|------------:|---------:|
| eo_large_arxiv_no_errors_finetuned |                    97.0739 |     516.967 | 0.021558 |
| Benchmark            | Split   |   Runtime in seconds |   Number of samples |   File size in KB |
|:---------------------|:--------|---------------------:|--------------------:|------------------:|
| ACL                  | test    |              1.27252 |                 462 |           56.0674 |
| arXiv.OCR            | test    |             21.6544  |                9360 |          882.581  |
| arXiv.pdftotext      | test    |             22.0294  |                9371 |          878.843  |
| doval                | test    |              1.88161 |                1000 |           81.6943 |
| Wiki                 | test    |             21.0617  |                9997 |          914.075  |
| Wiki.typos.no_spaces | test    |             19.6519  |                9997 |          775.755  |
| Wiki.typos           | test    |             21.9654  |                9997 |          913.895  |

| Model                                           |   Total runtime in seconds |   samples/s |      s/KB |
|:------------------------------------------------|---------------------------:|------------:|----------:|
| eo_large_arxiv_with_errors_temp_thres_optimized |                    109.517 |      458.23 | 0.0243214 |
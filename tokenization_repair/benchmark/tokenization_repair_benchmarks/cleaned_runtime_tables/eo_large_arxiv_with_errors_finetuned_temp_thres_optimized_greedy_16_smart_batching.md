| Benchmark            | Split   |   Runtime in seconds |   Number of samples |   File size in KB |
|:---------------------|:--------|---------------------:|--------------------:|------------------:|
| ACL                  | test    |              1.23192 |                 462 |           56.0674 |
| arXiv.OCR            | test    |             18.1604  |                9360 |          882.581  |
| arXiv.pdftotext      | test    |             18.6729  |                9371 |          878.843  |
| doval                | test    |              1.73035 |                1000 |           81.6943 |
| Wiki                 | test    |             18.7718  |                9997 |          914.075  |
| Wiki.typos.no_spaces | test    |             16.4964  |                9997 |          775.755  |
| Wiki.typos           | test    |             18.8683  |                9997 |          913.895  |

| Model                                                     |   Total runtime in seconds |   samples/s |      s/KB |
|:----------------------------------------------------------|---------------------------:|------------:|----------:|
| eo_large_arxiv_with_errors_finetuned_temp_thres_optimized |                    93.9321 |     534.258 | 0.0208603 |
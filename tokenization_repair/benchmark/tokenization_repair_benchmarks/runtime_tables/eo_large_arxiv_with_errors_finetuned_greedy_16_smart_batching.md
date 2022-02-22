| Benchmark            | Split   |   Runtime in seconds |   Number of samples |   File size in KB |
|:---------------------|:--------|---------------------:|--------------------:|------------------:|
| ACL                  | test    |              1.17103 |                 462 |           56.0674 |
| arXiv.OCR            | test    |             17.7448  |                9360 |          882.581  |
| arXiv.pdftotext      | test    |             17.7302  |                9371 |          878.843  |
| doval                | test    |              1.69516 |                1000 |           81.6943 |
| Wiki                 | test    |             17.9653  |                9997 |          914.075  |
| Wiki.typos.no_spaces | test    |             15.7667  |                9997 |          775.755  |
| Wiki.typos           | test    |             18.2653  |                9997 |          913.895  |

| Model                                |   Total runtime in seconds |   samples/s |      s/KB |
|:-------------------------------------|---------------------------:|------------:|----------:|
| eo_large_arxiv_with_errors_finetuned |                    90.3385 |     555.511 | 0.0200622 |
| Benchmark            | Split   |   Runtime in seconds |   Number of samples |   File size in KB |
|:---------------------|:--------|---------------------:|--------------------:|------------------:|
| ACL                  | test    |             115.682  |                 462 |           56.0674 |
| arXiv.OCR            | test    |            1590.09   |                9360 |          882.581  |
| arXiv.pdftotext      | test    |            1579.88   |                9371 |          878.843  |
| doval                | test    |              80.7252 |                1000 |           81.6943 |
| Wiki                 | test    |            1063.42   |                9997 |          914.075  |
| Wiki.typos.no_spaces | test    |             791.255  |                9997 |          775.755  |
| Wiki.typos           | test    |            1077.09   |                9997 |          913.895  |

| Model                               |   Total runtime in seconds |   samples/s |    s/KB |
|:------------------------------------|---------------------------:|------------:|--------:|
| nmt_large_arxiv_no_errors_finetuned |                    6298.14 |     7.96807 | 1.39868 |
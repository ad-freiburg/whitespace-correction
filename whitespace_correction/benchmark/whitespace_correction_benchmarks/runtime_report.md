| Task | Model | Input size | Runtime in seconds | Seq/s | kChar/s | MiB GPU memory | Mio. parameters | Precision | Batch size | Sorted | Device |
| :-- | --: | --: | --: | --: | --: | --: | --: | --: | --: | --: | --: |
| whitespace correction | eo_large_arxiv_with_errors | 3,500 sequences, 392,359 chars | 6.7 | 522.3 | 58.5 | 662 | 37.9 | fp32 | 16 | yes | NVIDIA GeForce RTX 2080 Ti, AMD EPYC 7502 32-Core Processor |
| whitespace correction | eo_large_arxiv_with_errors | 3,500 sequences, 392,359 chars | 4.6 | 763.9 | 85.6 | 556 | 37.9 | fp16 | 16 | yes | NVIDIA GeForce RTX 2080 Ti, AMD EPYC 7502 32-Core Processor |
| whitespace correction | eo_large_arxiv_with_errors | 3,500 sequences, 392,359 chars | 6.1 | 575.8 | 64.5 | 1,174 | 37.9 | fp32 | 32 | yes | NVIDIA GeForce RTX 2080 Ti, AMD EPYC 7502 32-Core Processor |
| whitespace correction | eo_large_arxiv_with_errors | 3,500 sequences, 392,359 chars | 4.0 | 878.4 | 98.5 | 966 | 37.9 | fp16 | 32 | yes | NVIDIA GeForce RTX 2080 Ti, AMD EPYC 7502 32-Core Processor |
| whitespace correction | eo_medium_arxiv_with_errors | 3,500 sequences, 392,359 chars | 4.0 | 882.2 | 98.9 | 592 | 19.0 | fp32 | 16 | yes | NVIDIA GeForce RTX 2080 Ti, AMD EPYC 7502 32-Core Processor |
| whitespace correction | eo_medium_arxiv_with_errors | 3,500 sequences, 392,359 chars | 3.1 | 1143.3 | 128.2 | 486 | 19.0 | fp16 | 16 | yes | NVIDIA GeForce RTX 2080 Ti, AMD EPYC 7502 32-Core Processor |
| whitespace correction | eo_medium_arxiv_with_errors | 3,500 sequences, 392,359 chars | 3.7 | 945.3 | 106.0 | 1,088 | 19.0 | fp32 | 32 | yes | NVIDIA GeForce RTX 2080 Ti, AMD EPYC 7502 32-Core Processor |
| whitespace correction | eo_medium_arxiv_with_errors | 3,500 sequences, 392,359 chars | 2.7 | 1313.6 | 147.3 | 878 | 19.0 | fp16 | 32 | yes | NVIDIA GeForce RTX 2080 Ti, AMD EPYC 7502 32-Core Processor |
| whitespace correction | eo_small_arxiv_with_errors | 3,500 sequences, 392,359 chars | 2.7 | 1277.3 | 143.2 | 550 | 9.5 | fp32 | 16 | yes | NVIDIA GeForce RTX 2080 Ti, AMD EPYC 7502 32-Core Processor |
| whitespace correction | eo_small_arxiv_with_errors | 3,500 sequences, 392,359 chars | 2.3 | 1529.2 | 171.4 | 444 | 9.5 | fp16 | 16 | yes | NVIDIA GeForce RTX 2080 Ti, AMD EPYC 7502 32-Core Processor |
| whitespace correction | eo_small_arxiv_with_errors | 3,500 sequences, 392,359 chars | 2.5 | 1392.3 | 156.1 | 1,044 | 9.5 | fp32 | 32 | yes | NVIDIA GeForce RTX 2080 Ti, AMD EPYC 7502 32-Core Processor |
| whitespace correction | eo_small_arxiv_with_errors | 3,500 sequences, 392,359 chars | 2.1 | 1693.8 | 189.9 | 836 | 9.5 | fp16 | 32 | yes | NVIDIA GeForce RTX 2080 Ti, AMD EPYC 7502 32-Core Processor |
| whitespace correction | nmt_large_char2char | 3,500 sequences, 392,359 chars | 965.3 | 3.6 | 0.4 | 768 | 44.2 | fp32 | 16 | yes | NVIDIA GeForce RTX 2080 Ti, AMD EPYC 7502 32-Core Processor |
| whitespace correction | nmt_large_char2char | 3,500 sequences, 392,359 chars | 804.0 | 4.4 | 0.5 | 668 | 44.2 | fp16 | 16 | yes | NVIDIA GeForce RTX 2080 Ti, AMD EPYC 7502 32-Core Processor |
| whitespace correction | nmt_large_char2char | 3,500 sequences, 392,359 chars | 930.6 | 3.8 | 0.4 | 1,344 | 44.2 | fp32 | 32 | yes | NVIDIA GeForce RTX 2080 Ti, AMD EPYC 7502 32-Core Processor |
| whitespace correction | nmt_large_char2char | 3,500 sequences, 392,359 chars | 750.1 | 4.7 | 0.5 | 1,146 | 44.2 | fp16 | 32 | yes | NVIDIA GeForce RTX 2080 Ti, AMD EPYC 7502 32-Core Processor |
| whitespace correction | nmt_medium_char2char | 3,500 sequences, 392,359 chars | 744.2 | 4.7 | 0.5 | 680 | 22.2 | fp32 | 16 | yes | NVIDIA GeForce RTX 2080 Ti, AMD EPYC 7502 32-Core Processor |
| whitespace correction | nmt_medium_char2char | 3,500 sequences, 392,359 chars | 677.8 | 5.2 | 0.6 | 580 | 22.2 | fp16 | 16 | yes | NVIDIA GeForce RTX 2080 Ti, AMD EPYC 7502 32-Core Processor |
| whitespace correction | nmt_medium_char2char | 3,500 sequences, 392,359 chars | 716.4 | 4.9 | 0.5 | 1,254 | 22.2 | fp32 | 32 | yes | NVIDIA GeForce RTX 2080 Ti, AMD EPYC 7502 32-Core Processor |
| whitespace correction | nmt_medium_char2char | 3,500 sequences, 392,359 chars | 632.4 | 5.5 | 0.6 | 1,058 | 22.2 | fp16 | 32 | yes | NVIDIA GeForce RTX 2080 Ti, AMD EPYC 7502 32-Core Processor |
| whitespace correction | nmt_small_char2char | 3,500 sequences, 392,359 chars | 592.9 | 5.9 | 0.7 | 656 | 10.6 | fp32 | 16 | yes | NVIDIA GeForce RTX 2080 Ti, AMD EPYC 7502 32-Core Processor |
| whitespace correction | nmt_small_char2char | 3,500 sequences, 392,359 chars | 560.1 | 6.2 | 0.7 | 534 | 10.6 | fp16 | 16 | yes | NVIDIA GeForce RTX 2080 Ti, AMD EPYC 7502 32-Core Processor |
| whitespace correction | nmt_small_char2char | 3,500 sequences, 392,359 chars | 570.7 | 6.1 | 0.7 | 1,260 | 10.6 | fp32 | 32 | yes | NVIDIA GeForce RTX 2080 Ti, AMD EPYC 7502 32-Core Processor |
| whitespace correction | nmt_small_char2char | 3,500 sequences, 392,359 chars | 529.1 | 6.6 | 0.7 | 1,012 | 10.6 | fp16 | 32 | yes | NVIDIA GeForce RTX 2080 Ti, AMD EPYC 7502 32-Core Processor |

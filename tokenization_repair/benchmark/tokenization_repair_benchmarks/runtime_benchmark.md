| Task                | Model                       | Input size                     | Runtime in seconds |  Seq/s |    Char/s | Batch size | Sorted | Device                                                               |
|:--------------------|:----------------------------|:-------------------------------|-------------------:|-------:|----------:|-----------:|:-------|:---------------------------------------------------------------------|
| tokenization repair | eo_large_arxiv_with_errors  | 1,400 sequences, 150,885 chars |              3.369 | 415.60 |  44791.52 |         16 | yes    | NVIDIA GeForce GTX 1080 Ti, Intel(R) Core(TM) i7-9750H CPU @ 2.60GHz |
| tokenization repair | eo_medium_arxiv_with_errors | 1,400 sequences, 150,885 chars |              2.009 | 696.99 |  75118.19 |         16 | yes    | NVIDIA GeForce GTX 1080 Ti, Intel(R) Core(TM) i7-9750H CPU @ 2.60GHz |
| tokenization repair | eo_small_arxiv_with_errors  | 1,400 sequences, 150,885 chars |              1.460 | 958.93 | 103349.19 |         16 | yes    | NVIDIA GeForce GTX 1080 Ti, Intel(R) Core(TM) i7-9750H CPU @ 2.60GHz |
| tokenization repair | nmt_small_char2char         | 1,400 sequences, 150,885 chars |            270.166 |   5.18 |    558.49 |         16 | yes    | NVIDIA GeForce GTX 1080 Ti, Intel(R) Core(TM) i7-9750H CPU @ 2.60GHz |
| tokenization repair | nmt_medium_char2char        | 1,400 sequences, 150,885 chars |            341.206 |   4.10 |    442.21 |         16 | yes    | NVIDIA GeForce GTX 1080 Ti, Intel(R) Core(TM) i7-9750H CPU @ 2.60GHz |
| tokenization repair | nmt_large_char2char         | 1,400 sequences, 150,885 chars |            470.617 |   2.97 |    320.61 |         16 | yes    | NVIDIA GeForce GTX 1080 Ti, Intel(R) Core(TM) i7-9750H CPU @ 2.60GHz |
| tokenization_repair | wordsegment                 | 1,400 sequences, 150,885 chars |            111.600 |   12.5 |   1352.02 |          1 | no     | Intel(R) Core(TM) i7-9750H CPU @ 2.60GHz                             |


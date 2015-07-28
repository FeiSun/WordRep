# Learning Word Representations by Jointly Modeling Syntagmatic and Paradigmatic Relations


## Introduction

This is a toolkit developed for learning the word representations. 
PDC and HDC are two unsupervised learning algorithms for word representations using both syntagmatic and paradigmatic relations via a joint training objective.

```
Contact: Fei Sun, Institute Of Computing Technology, ofey.sunfei@gmail.com, 
Project page: http://ofey.me/projects/wordrep
```

## Usage

**Requirements**

To complile the souce codes, some external packages are required

* C++11
* Eigen
* OpenMP (for multithread)

**Input**

Each line of the input file represents a document in corpus.

```
... The cat sat on the mat. ...
... The quick brown fox jumps over the lazy dog. ...
```

**Run**

```shell
./w2v -train data.txt -word_output vec.txt -size 200 -window 5 -subsample 1e-4 -negative 5 -model pdc -binary 0 -iter 5
```

- -train, the input file of the corpus, each line a document;
- -word_output, the output file of the word embeddings;
- -binary, whether saving the output file in binary mode; the default is 0 (off);
- -word_size, the dimension of word embeddings; the default is 100;
- -doc_size, the dimension of word embeddings; the default is 100;
- -window, max skip length between words; default is 5;
- -negative, the number of negative samples used in negative sampling; the deault is 5;
- -subsample, parameter for subsampling; default is 1e-4;
- -threads, the total number of threads used; the default is 1.
- -alpha, the starting learning rate; default is 0.025 for HDC and 0.05 for PDC; 
- -model, model used to learn the word embeddings; default is Parallel Document Context model(pdc) (use hdc for Hierarchical Document Context model)
- -min-count, the threshold for occurrence of words; default is 5;
- -iter, the number of iterations; default is 5;


## Citation

```TeX
@inproceedings{Sun:2011:DBC:2009916.2009952,
author = {Sun, Fei and Song, Dandan and Liao, Lejian},
title = {DOM based content extraction via text density},
booktitle = {Proceedings of the 34th international ACM SIGIR conference on Research and development in Information Retrieval},
series = {SIGIR '11},
year = {2011},
isbn = {978-1-4503-0757-4},
location = {Beijing, China},
pages = {245--254},
numpages = {10},
url = {http://doi.acm.org/10.1145/2009916.2009952},
doi = {10.1145/2009916.2009952},
acmid = {2009952},
publisher = {ACM},
address = {New York, NY, USA},
keywords = {composite text density, content extraction, densitysum, text density},
}
```

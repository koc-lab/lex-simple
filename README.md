# lex-simple
Repo for the paper Unsupervised Simplification of Legal Texts https://arxiv.org/pdf/2209.00557

## Dataset
We have gathered a new dataset for the goal of legal text simplification. To that aim, we have selected 200 random legal sentences from the CaseLaw Access project of Harward Law School. Then, by collaborating with the faculty and the students of Bilkent Law School, we produced 3 different simplified reference files for these 200 sentences. We hope that this dataset can serve as a benchmark for future legal text simplification studies.

## Code
In order to run the algorithm proposed in the paper, run the following command. Python 3.6 or above is required.
```
python3 scripts/n_evaluate_asl.py
```

## Evaluation

After gathering the text outputs, run 
```
python3 eval.py
```

you may need to install the readability and easse packages. For readability, run
```
pip install readability
```
or 
```
pip install https://github.com/andreasvc/readability/tarball/master
```

For easse, follow the guides in https://github.com/feralvam/easse

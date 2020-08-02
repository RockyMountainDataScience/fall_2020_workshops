# Workshop 1 - Machine Learning Fundamentals I

In the first workshop on machine learning (ML), we will cover the following
topics:

- A brief Introduction to ML
- ML Lifecycle
- Types of ML problems (e.g. Regression, Classification)
- Types of Data (Research Examples)

This is given as a presentation that you can find [here](workshop_1.slides.html)

To compile the slides yourself, first create and activate the conda environment with:
```
conda env create -f environment.yml
conda activate rmds-f2020-ws1
```

Then, install and enable the RISE jupyter notebook extension with:
```
jupyter-nbextension install rise --py --sys-prefix
jupyter nbextension enable rise --py --sys-prefix
```

Finally, compile the slides into an html file with:

```
jupyter nbconvert --to slides workshop_1.ipynb --post serve
```
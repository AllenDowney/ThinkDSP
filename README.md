# ThinkDSP

*Think DSP* is an introduction to Digital Signal Processing in Python.

[Order *Think DSP* from Amazon.com](http://amzn.to/1naaUCN).

[Download the first edition in PDF](http://greenteapress.com/thinkdsp/thinkdsp.pdf).

[Read the first edition in HTML](http://greenteapress.com/thinkdsp/html/index.html).

The premise of this book (and the other books in the Think X series) is that if you know how to program, you can use that skill to learn other things. I am writing this book because I think the conventional approach to digital signal processing is backward: most books (and the classes that use them) present the material bottom-up, starting with mathematical abstractions like phasors.

With a programming-based approach, I can go top-down, which means I can present the most important ideas right away. By the end of the first chapter, you can decompose a sound into its harmonics, modify the harmonics, and generate new sounds.

Think DSP is a Free Book. It is available under the [Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International](https://creativecommons.org/licenses/by-nc-sa/4.0/), which means that you are free to copy, distribute, and modify it, as long as you attribute the work and don't use it for commercial purposes.

## Which repository?

*Think DSP* has two GitHub homes. Use this map to pick the right one:

1. **Python library** — **this repo**: optional `pip install think-dsp` if you want the package; chapter notebooks can instead download `thinkdsp.py` (see Install below).
2. **First edition (PDF/HTML) and archived `code/` tree** — **this repo**: published book links above; old layout on the [`edition-1`](https://github.com/AllenDowney/ThinkDSP/tree/edition-1) branch.
3. **Second edition draft (book-as-notebooks)** — [**ThinkDSP2**](https://github.com/AllenDowney/ThinkDSP2): Jupyter Book WIP where each chapter notebook includes the text, code, and exercises.
4. **Which notebooks to run?**
   - **Separate chapter code and exercises** (classic style) — **this repo**, under `nb/`: e.g. `chap01.ipynb` for examples/exercises and `chap01soln.ipynb` for solutions.
   - **One notebook per chapter** (full chapter text + code + exercises) — **ThinkDSP2**: use [`notebooks/`](https://github.com/AllenDowney/ThinkDSP2/tree/main/notebooks) without solutions, or [`soln/`](https://github.com/AllenDowney/ThinkDSP2/tree/main/soln) with solutions.

## Install

How you get `thinkdsp` depends on what you are doing:

1. **Running the chapter notebooks (default)** — you do **not** need to install the package. Notebooks download `thinkdsp.py` (and any missing data files from `data/`) into the working directory when you run them. Install only the notebook stack if you are working locally:

   ```bash
   pip install -r requirements.txt
   ```

2. **Using the library in your own code (optional)** — install from PyPI:

   ```bash
   pip install think-dsp
   ```

   Then `import thinkdsp` works without a download cell.

3. **Developing this repository** — install an editable checkout plus dev tools (what CI uses):

   ```bash
   pip install -r requirements-dev.txt
   ```

   Or with conda: `make create_environment_dev` (also does `pip install -e .`).

Chapter notebooks live under `nb/`. Shared datasets (CSV, WAV) live under `data/`.

Here's a notebook that previews what you will see in Chapter 1:

* [chap01.ipynb](https://colab.research.google.com/github/AllenDowney/ThinkDSP/blob/master/nb/chap01.ipynb)

And if you want to see where we are headed, here's a preview of Chapter 10:

* [chap10.ipynb](https://colab.research.google.com/github/AllenDowney/ThinkDSP/blob/master/nb/chap10.ipynb)


## Running the code

Most of the code for this book is in Jupyter notebooks.
If you are not familiar with Jupyter, you can run a tutorial by [clicking here](https://jupyter.org/try).
To run the ThinkDSP code, you have several options:

1. **Google Colab** — Best for a quick start: no local install, works in a browser, free.
2. **Conda on your computer** — Best for a stable local setup and offline work.
   Downsides: larger download and a bit more setup than Colab.
3. **Poetry on your computer** — Best if you already use Poetry / prefer a project-local virtualenv.

The following sections explain these options in detail.

Note: I have heard from a few people who tried to run the code in Spyder.  Apparently there were problems, so I don't recommend it.

### Option 1: Run on Colab

Most of the notebooks in this repository so run on Colab. If you find one that doesn't, let me know and I will update it.

You can open any of them by clicking on the links below.  If you want to modify and save any of them, you can use Colab to save a copy in a Google Drive or your own GitHub repo, or on your computer.

* [chap01.ipynb](https://colab.research.google.com/github/AllenDowney/ThinkDSP/blob/master/nb/chap01.ipynb)
* [chap01soln.ipynb](https://colab.research.google.com/github/AllenDowney/ThinkDSP/blob/master/nb/chap01soln.ipynb)
* [chap02.ipynb](https://colab.research.google.com/github/AllenDowney/ThinkDSP/blob/master/nb/chap02.ipynb)
* [chap02soln.ipynb](https://colab.research.google.com/github/AllenDowney/ThinkDSP/blob/master/nb/chap02soln.ipynb)
* [chap03.ipynb](https://colab.research.google.com/github/AllenDowney/ThinkDSP/blob/master/nb/chap03.ipynb)
* [chap03soln.ipynb](https://colab.research.google.com/github/AllenDowney/ThinkDSP/blob/master/nb/chap03soln.ipynb)
* [chap04.ipynb](https://colab.research.google.com/github/AllenDowney/ThinkDSP/blob/master/nb/chap04.ipynb)
* [chap04soln.ipynb](https://colab.research.google.com/github/AllenDowney/ThinkDSP/blob/master/nb/chap04soln.ipynb)
* [chap05.ipynb](https://colab.research.google.com/github/AllenDowney/ThinkDSP/blob/master/nb/chap05.ipynb)
* [chap05soln.ipynb](https://colab.research.google.com/github/AllenDowney/ThinkDSP/blob/master/nb/chap05soln.ipynb)
* [chap06.ipynb](https://colab.research.google.com/github/AllenDowney/ThinkDSP/blob/master/nb/chap06.ipynb)
* [chap06soln.ipynb](https://colab.research.google.com/github/AllenDowney/ThinkDSP/blob/master/nb/chap06soln.ipynb)
* [chap07.ipynb](https://colab.research.google.com/github/AllenDowney/ThinkDSP/blob/master/nb/chap07.ipynb)
* [chap07soln.ipynb](https://colab.research.google.com/github/AllenDowney/ThinkDSP/blob/master/nb/chap07soln.ipynb)
* [chap08.ipynb](https://colab.research.google.com/github/AllenDowney/ThinkDSP/blob/master/nb/chap08.ipynb)
* [chap08soln.ipynb](https://colab.research.google.com/github/AllenDowney/ThinkDSP/blob/master/nb/chap08soln.ipynb)
* [chap09.ipynb](https://colab.research.google.com/github/AllenDowney/ThinkDSP/blob/master/nb/chap09.ipynb)
* [chap09soln.ipynb](https://colab.research.google.com/github/AllenDowney/ThinkDSP/blob/master/nb/chap09soln.ipynb)
* [chap10.ipynb](https://colab.research.google.com/github/AllenDowney/ThinkDSP/blob/master/nb/chap10.ipynb)
* [chap10soln.ipynb](https://colab.research.google.com/github/AllenDowney/ThinkDSP/blob/master/nb/chap10soln.ipynb)
* [chap11.ipynb](https://colab.research.google.com/github/AllenDowney/ThinkDSP/blob/master/nb/chap11.ipynb)
* [chap11soln.ipynb](https://colab.research.google.com/github/AllenDowney/ThinkDSP/blob/master/nb/chap11soln.ipynb)


### Option 2: Install Python+Jupyter with Conda

First, download the files from this repository.  If you are a Git user, you can run

```
git clone --depth 1 https://github.com/AllenDowney/ThinkDSP.git
```

Otherwise you can [download this Zip file](https://github.com/AllenDowney/ThinkDSP/archive/master.zip) and unzip it.
Either way, you should end up with a directory called `ThinkDSP`.

Now, if you don't already have Jupyter, I highly recommend installing Anaconda, which is a Python distribution that contains everything you need to run the ThinkDSP code.  It is easy to install on Windows, Mac, and Linux, and because it does a
user-level install, it will not interfere with other Python installations.

[Information about installing Anaconda is here](https://www.anaconda.com/distribution/).

If you have the choice of Python 2 or 3, choose Python 3.

There are two ways to get the packages you need for ThinkDSP.  You can install them by hand or create a Conda environment.

To install them by hand run

```
conda install jupyter numpy scipy pandas matplotlib seaborn
```

Or, to create a conda environment, run

```
cd ThinkDSP
conda env create -f environment.yml
conda activate ThinkDSP
```


### Option 3: Use Poetry locally

First, download the files from this repository.  If you are a Git user, you can run

```
git clone --depth 1 https://github.com/AllenDowney/ThinkDSP.git
```

Then, assuming you have [poetry](https://python-poetry.org) installed on your machine, run

```
cd ThinkDSP
poetry install
```

to install the libraries you need in a virtual environment.  To activate the environment, run

```
poetry shell
```

Then you can run Jupyter.


## Run Jupyter 

To start Jupyter, run:

```
jupyter notebook
```

Jupyter should launch your default browser or open a tab in an existing browser window.
If not, the Jupyter server should print a URL you can use.  For example, when I launch Jupyter, I get

```
~/ThinkComplexity2$ jupyter notebook
[I 10:03:20.115 NotebookApp] Serving notebooks from local directory: /home/downey/ThinkDSP
[I 10:03:20.115 NotebookApp] 0 active kernels
[I 10:03:20.115 NotebookApp] The Jupyter Notebook is running at: http://localhost:8888/
[I 10:03:20.115 NotebookApp] Use Control-C to stop this server and shut down all kernels (twice to skip confirmation).
```

In this case, the URL is [http://localhost:8888](http://localhost:8888).
When you start your server, you might get a different URL.
Whatever it is, if you paste it into a browser, you should see a home page with a list of directories.

Click on `nb` to open the folder with the notebooks, then click on one of the notebooks (with the .ipynb extension).

Select the cell with the import statements and press "Shift-Enter" to run the code in the cell.
If it works and you get no error messages, **you are all set**.

If you get error messages about missing packages, you can install the packages you need using your
package manager, or install Anaconda.

If you run into problems with these instructions, let me know and I will make corrections.  Good luck!



## Freesound

Special thanks to Freesound (http://freesound.org), which is the source of many of the
sound samples I use in this book, and to the Freesound users who
uploaded those sounds.  I include some of their wave files in
the GitHub repository for this book, using the original file
names, so it should be easy to find their sources.

Unfortunately, most Freesound users don't make their real names
available, so I can only thank them using their user names.  Samples
used in this book were contributed by Freesound users: iluppai,
wcfl10, thirsk, docquesting, kleeb, landup, zippi1, themusicalnomad,
bcjordan, rockwehrmann, marchascon7, jcveliz.  Thank you all!

Here are links to the sources:

http://www.freesound.org/people/iluppai/sounds/100475/

http://www.freesound.org/people/wcfl10/sounds/105977/

http://www.freesound.org/people/Thirsk/sounds/120994/

http://www.freesound.org/people/ciccarelli/sounds/132736/

http://www.freesound.org/people/Kleeb/sounds/180960/

http://www.freesound.org/people/zippi1/sounds/18871/

http://www.freesound.org/people/themusicalnomad/sounds/253887/

http://www.freesound.org/people/bcjordan/sounds/28042/

http://www.freesound.org/people/rockwehrmann/sounds/72475/

http://www.freesound.org/people/marcgascon7/sounds/87778/

http://www.freesound.org/people/jcveliz/sounds/92002/

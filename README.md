# ThinkDSP

LaTeX source and Python code for _Think DSP: Digital Signal Processing in Python_, by Allen B. Downey.

The premise of this book (and the other books in the _Think X_ series) is that if you know how to program,
you can use that skill to learn other things.  I am writing this book because I think the conventional
approach to digital signal processing is backward: most books (and the classes that use them) present
the material bottom-up, starting with mathematical abstractions like phasors.

With a programming-based approach, I can go top-down, which means I can present the most important
ideas right away.  By the end of the first chapter, you can break down a sound into its harmonics, modify the harmonics, and generate new sounds.

## Running the code

Most of the code for this book is in Jupyter notebooks.
If you are not familiar with Jupyter, you can run a tutorial by [clicking here](https://jupyter.org/try).  Then select "Try Classic Notebook".  It will open a notebook with instructions for getting started.

To run the ThinkDSP code, you have two options:

1. The simplest option is to run the code on Binder.  The drawback is that the notebooks run in a temporary environment; if you leave a notebook idle for a while, the temporary environment goes away and you lose any changes you made.

2. The other option is to install Python, Jupyter, and the other packages you need on your computer, and download my code from GitHub.

The following two sections explain these options in detail.

Note: I have heard from a few people who tried to run the code in Spyder.  Apparently there were problems, so I don't recommend it.


### Option 1: Run on Binder

To run the code for this book on Binder, press this button:

[![Binder](http://mybinder.org/badge.svg)](http://mybinder.org/repo/AllenDowney/ThinkDSP)

It takes a minute or so to start up, but then you should see the Jupyter home page with a list of files.  Click on `code` to open the folder with the notebooks, then click on one of the notebooks (with the .ipynb extension).


### Option 2: Install Python+Jupyter

First, download the files from this repository.  If you are a Git user, you can run

```
git clone --depth 1 https://github.com/AllenDowney/ThinkDSP.git
```

Otherwise you can [download this Zip file](https://github.com/AllenDowney/ThinkDSP/archive/master.zip) and unzip it.
Either way, you should end up with a directory called `ThinkDSP`.

Now, if you don't already have Jupyter, I highly recommend installing Anaconda, which is a Python distribution that contains everything you need to run the ThinkDSP code.  It is easy to install on Windows, Mac, and Linux, and because it does a
user-level install, it will not interfere with other Python installations.

[Information about installing Anaconda is here](https://www.anaconda.com/distribution/).

If you have the choice of Python 2 or 3, choose 3.

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
Whatever it is, if you paste it into a browser, you should should see a home page with a list of directories.

Click on `code` to open the folder with the notebooks, then click on one of the notebooks (with the .ipynb extension).

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

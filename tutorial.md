## Tutorial: Introduction to Digital Signal Processing

_Intro to DSP_ is a half-day tutorial that uses material from _Think DSP_ to 
introduce the fundamental ideas of Digital Signal Processing, in particular
spectral analysis: the idea that a signal that varies in time can be expressed
as a sum of frequency components, and that operations on signals can be
represented equivalently in terms of time or frequency.

If you don't quite understand what that means, this tutorial is for you!

Important: **Please bring headphones or earbuds!**  Many of the exercises in this tutorial
make noise.  If people don't have headphones, it's going to be chaos!

### Installation instructions

To prepare for this tutorial, you have two options:

1. Install Jupyter on your laptop and download my code from Git.

2. Run the Jupyter notebook on a virtual machine on Binder.

I'll provide instructions for both, but here's the catch: if everyone chooses Option 2, 
the wireless network will fail and no one will be able to do the hands-on part of the workshop.

So, I strongly encourage you to try Option 1 and only resort to Option 2 if you can't get Option 1 working.

### Option 1A: If you already have Jupyter installed.

To do the exercises, you need Python 2 or 3 with NumPy, SciPy, and matplotlib.
If you are not sure whether you have those modules already, the easiest way to
check is to run my code and see if it works.

Code for this workshop is in a Git repository on Github.  
If you have a Git client installed, you should be able to download it by running:

    git clone https://github.com/AllenDowney/ThinkDSP.git

It should create a directory named `ThinkDSP`.
Otherwise you can download the repository in [this zip file](https://github.com/AllenDowney/ThinkDSP/zipball/gh-pages).

To start Jupyter, run:

    cd ThinkDSP/code
    jupyter notebook

Jupyter should launch your default browser or open a tab in an existing browser window.
If not, the Jupyter server should print a URL you can use.  For example, when I launch Jupyter, I get

    ~/ThinkDSP$ jupyter notebook
    [I 10:03:20.115 NotebookApp] Serving notebooks from local directory: /home/downey/ThinkDSP
    [I 10:03:20.115 NotebookApp] 0 active kernels 
    [I 10:03:20.115 NotebookApp] The Jupyter Notebook is running at: http://localhost:8888/
    [I 10:03:20.115 NotebookApp] Use Control-C to stop this server and shut down all kernels (twice to skip confirmation).

In this case, the URL is [http://localhost:8888](http://localhost:8888).  
When you start your server, you might get a different URL.
Whatever it is, if you paste it into a browser, you should should see a home page with a list of the
notebooks in the repository.

Click on `chap01.ipynb`.  It should open the notebook for Chapter 1.

Select the cell with the import statements and press "Shift-Enter" to run the code in the cell.
If it works and you get no error messages, **you are all set**.  

If you get error messages about missing packages, you can install the packages you need using your
package manager, or try Option 1B and install Anaconda.


### Option 1B: If you don't already have Jupyter.

I highly recommend installing Anaconda, which is a Python distribution that contains everything
you need for the workshop.  It is easy to install on Windows, Mac, and Linux, and because it does a
user-level install, it will not interfere with other Python installations.

[Information about installing Anaconda is here](http://docs.continuum.io/anaconda/install.html).

When you install Anaconda, you should get Jupyter by default, but if not, run

    conda install jupyter

Then go to Option 1A to make sure you can run my code.

If you don't want to install Anaconda, 
[you can see some other options here](http://jupyter.readthedocs.io/en/latest/install.html).


### Option 2: only if Option 1 failed.

You can run my notebook in a virtual machine on Binder. To launch the VM, press this button:

 [![Binder](http://mybinder.org/badge.svg)](http://mybinder.org:/repo/allendowney/thinkdsp)

You should see a home page with a list of the notebooks in the repository.

If you want to try the exercises, open 'chap01.ipynb`. If you just want to see the answers, open `chap01soln.ipynb`.
Either way, you should be able to run the notebooks in your browser and try out the examples.  

However, be aware that the virtual machine you are running is temporary.
If you leave it idle for more than an hour or so, it will disappear along with any work you have done.

Special thanks to the generous people who run Binder, which makes it easy to share and reproduce computation.

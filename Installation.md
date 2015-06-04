# Introduction #

If you don't know python and just want to process your PALM data with a simple interface, use our standalone Windows version. If you want more control or a different operating system, you'll need [python](http://www.python.org/) running on your machine.

## Short summary for Windows users: ##

Download the latest version of [palm3d\_Windows\_vX.zip](http://code.google.com/p/palm3d/downloads/list) (X will be the version number). Unzip, open the 'palm3d\_Windows' folder, and double-click 'palm3d.wsf' to start the graphical interface.


## Long version for everyone else: ##

palm3d is written in [python](http://www.python.org/), and depends on third-party packages: [numpy](http://numpy.scipy.org/), [scipy](http://www.scipy.org/), [matplotlib](http://matplotlib.sourceforge.net/), [ipython](http://ipython.scipy.org/moin/), and, optionally, the [Python Imaging Library](http://www.pythonware.com/products/pil/) for opening [TIF](http://en.wikipedia.org/wiki/Tagged_Image_File_Format) images. You'll need this software installed for palm3d to run. This is easy in Ubuntu and Windows, and probably not too bad on other Linux flavors, or Mac.

If you install python yourself, you'll also need a very basic [working knowledge](http://docs.python.org/tutorial/) of python: how to put a [module](http://docs.python.org/tutorial/modules.html) somewhere python can find it, how to [run a script](http://docs.python.org/faq/windows.html#how-do-i-run-a-python-program-under-windows) in python, and how to [edit](http://docs.python.org/library/idle.html) VERY simple python code.

After you've installed python, numpy, scipy, matplotlib, and ipython, [get the most recent version](http://code.google.com/p/palm3d/downloads/list) of 'palm\_3d.py' and 'palm\_gui.py'. Put 'palm\_3d.py' somewhere python can import it. Put 'palm\_gui.py' anywhere you like, and run it to start a graphical interface to 'palm\_3d.py'. The hardest part: 'palm\_gui.py' needs to know how to start ipython. I don't know how your operating system does this, so unless you're using one of the operating systems I've tested (or are very lucky), you'll have to edit the first function in 'palm\_gui.py', called 'ipython\_run'. If you don't know how to do this or find it annoying, use our standalone Windows version.

# Details #

In theory, palm3d should run wherever python runs. In practice, we've only tested palm3d on Windows XP, Windows 7, and Ubuntu 10.04, for certain versions of our third-party dependencies.

## This is what we've tested on Windows XP and 7: ##
  * [Python 2.6.1](http://www.python.org/ftp/python/2.6.6/python-2.6.6.msi)
  * [Numpy 1.4.1](http://sourceforge.net/projects/numpy/files/NumPy/1.4.1/numpy-1.4.1-win32-superpack-python2.6.exe/download)
  * [Scipy 0.7.2](http://sourceforge.net/projects/scipy/files/scipy/0.7.2/scipy-0.7.2-win32-superpack-python2.6.exe/download)
  * [Matplotlib 0.99.1](http://sourceforge.net/projects/matplotlib/files/matplotlib/matplotlib-0.99.1/matplotlib-0.99.1.win32-py2.6.exe/download)
  * [Ipython 0.9.1](http://ipython.scipy.org/dist/0.9.1/ipython-0.9.1.win32-setup.exe)
  * [PIL 1.1.6](http://effbot.org/downloads/PIL-1.1.6.win32-py2.6.exe) (we don't use .tif files much, so testing is sparse here)

I suspect palm3d would work with any recent versions of these packages, but just to be safe, I've linked to downloads of the versions we use. Download and install each file in the order above. If you don't want to install anything on your system, you should use our portable, no-install-required version of [palm3d for Windows](http://code.google.com/p/palm3d/downloads/list). This is actually a [standalone distribution](http://www.portablepython.com/) of python including all third-party dependencies. Try doing _that_ with Matlab!

## This is what we've tested on Ubuntu 10.04: ##
  * Python 2.6.5
  * Numpy 1.3.0
  * Scipy 0.7.0
  * Matplotlib 0.99.1.1
  * Ipython 0.10
  * PIL 1.1.7 (we don't use .tif files much, so testing is sparse here)

We installed all these packages through Ubuntu's [package manager](https://help.ubuntu.com/community/SynapticHowto), which made things very easy.

I'd be very happy if there was a .deb file to install palm3d directly in Ubuntu without any worry about dependencies, but I don't know how to make a .deb file yet. Any volunteers?
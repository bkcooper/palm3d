# Taking calibration data #
One of the big differences between our PALM code and most other PALM codes is that we calculate a glowing molecule's position by comparing to measured calibration images. This is especially important if you want to use a [water immersion objective](http://www.ncbi.nlm.nih.gov/pubmed/15369482), which (in my experience) has a different [point-spread function](http://en.wikipedia.org/wiki/Point_spread_function) every time you change your sample:

![https://sites.google.com/site/palm3ddocumentation/home/variation.png](https://sites.google.com/site/palm3ddocumentation/home/variation.png)

A nice side effect of our calibration-based approach is it's very aberration-tolerant. This means the palm3d code works for a wide variety of microscopes, and you don't have to make your imaging hardware perfect to match some arbitrary mathematical model. The drawback is that you have to take calibration data, but in practice this isn't hard.

Put a small glowing particle on your coverslip (we use [100 nanometer gold beads](http://www.microspheres-nanospheres.com/Microspheres/Inorganic/Metals/Gold.htm), which fluoresce under 561 nm illumination, don't bleach, and give a nice bright signal). Move your sample in small steps in the axial ('z') direction (we typically move 50 nanometers per step, from 2 microns negative defocus to 2 microns positive defocus). At each axial position, take one or more pictures of the glowing particle (perhaps 10 pictures per position), and save them as a stack of  [raw binary](http://rsbweb.nih.gov/ij/docs/menus/file.html#saveas) images, [formatted](http://code.google.com/p/palm3d/wiki/TakingData#Footnotes)<sup>1</sup> as unsigned 16-bit little-endian (Intel byte order) integers, with the '.dat' file extension. Repeat this process one or more times, and put the resulting files in a folder called 'calibration' in the experiment directory:

![https://sites.google.com/site/palm3ddocumentation/home/calfolder.png](https://sites.google.com/site/palm3ddocumentation/home/calfolder.png)

Make a note of what you did: step size, calibration image shape, images per position, and the number of repetitions. Whatever direction you moved your stage in while taking this data is now the positive 'z' direction.

There's a bit of a balancing act here. You want to take this calibration data quickly enough so that thermal drift isn't very large, but you want to use long exposure times or many images so you can get the best signal-to-noise ratio possible. We repeat the process to check if thermal drift has occurred; the images in the first and second acquisitions should look the same. If they don't, your stage isn't repeatable enough, possibly due to thermal drift.

If you change your sample and you're using a water-immersion objective, you probably changed your point-spread function too, and should retake your calibration stack.

# Taking PALM data the simple way #
Our graphical interface assumes your PALM data is in a particular format and stored a particular way, to simplify processing. If you're PALM-imaging a thin sample that fits entirely in your microscope's localization volume, then this structure is very simple:

![http://sites.google.com/site/palm3ddocumentation/home/SimpleExperiment.png](http://sites.google.com/site/palm3ddocumentation/home/SimpleExperiment.png)

Take your PALM data in several 'acquisitions' (which can be processed in parallel). Save each acquisition as a stack of  [raw binary](http://rsbweb.nih.gov/ij/docs/menus/file.html#saveas) images, [formatted](http://code.google.com/p/palm3d/wiki/TakingData#Footnotes)<sup>1</sup> as unsigned 16-bit little-endian (Intel byte order) integers, with the '.dat' file extension. Put each acquisition in its own folder in the experiment directory, and name the folders so they all start with the same prefix ('acquisition`_`', in this example). Put your calibration data folder in the experiment directory too.

If your sample is thicker, and you have to move your sample in the z-direction to PALM image the whole volume, things are more complicated. See below!

# Taking PALM data the flexible way #
## Drift tracking is crucial ##

3D PALM images take a loooong time to acquire, so you have to compensate for thermal drift or else your images are garbage. We measure sample drift by tracking a bright ['fiducial' particle](http://en.wikipedia.org/wiki/Fiduciary_marker#Medical_Imaging). We use [100 nanometer gold beads](http://www.microspheres-nanospheres.com/Microspheres/Inorganic/Metals/Gold.htm), which fluoresce under 561 nm illumination, don't bleach, and give a nice bright signal for 3D drift tracking:

![http://sites.google.com/site/palm3ddocumentation/home/TrackingParticleInFocus.jpg](http://sites.google.com/site/palm3ddocumentation/home/TrackingParticleInFocus.jpg)

## ...but drift tracking is tricky in thick samples ##
For samples thicker than a few microns, we take PALM data one slice at a time. For part of the acquisition, the fiducial is in focus:

![http://sites.google.com/site/palm3ddocumentation/home/InFocus.jpg](http://sites.google.com/site/palm3ddocumentation/home/InFocus.jpg)

but not for the whole acquisition. When we take data deep in our samples, the fiducial can be very out of focus:

![http://sites.google.com/site/palm3ddocumentation/home/OutOfFocus.jpg](http://sites.google.com/site/palm3ddocumentation/home/OutOfFocus.jpg)

An out-of-focus fiducial is dim and blurry, and frequently invisible, useless for measuring drift:

![http://sites.google.com/site/palm3ddocumentation/home/TrackingParticleOutOfFocus.jpg](http://sites.google.com/site/palm3ddocumentation/home/TrackingParticleOutOfFocus.jpg)

## So we use 'jump-tracking' ##

We haven't found a good way to scatter multiple fiducials at different depths in our samples, so we use another way to measure drift. Our method relies on [precise, repeatable](http://code.google.com/p/palm3d/wiki/TakingData#Footnotes)<sup>2</sup> motion of our sample in the axial (z) direction with a piezoelectric positioner:

  1. Find an interesting sample to image.
  1. Sprinkle gold fiducials on the sample until at least one fiducial sticks in the field of view.
  1. Focus our microscope on the fiducial, and call this position 'z=0'.
  1. Move the positioner to the depth in the sample we want to image (z=3 microns, perhaps).
  1. Take a small amount of data, quickly enough that no serious thermal drift occurs (perhaps 200 images).
  1. Jump our sample back to z=0, so the tracking particle is back in focus.
  1. Take some pictures of the tracking particle to measure drift (say, 20 images)
  1. Repeat steps 4-7 until we have a good amount of data to process (perhaps 50 repetitions).

Illustrated graphically:

http://sites.google.com/site/palm3ddocumentation/home/Acquisition.jpg?height=276&width=400

After one pass through steps 4-7, you might have data that looks like this:

![http://sites.google.com/site/palm3ddocumentation/home/TwoFiles.png](http://sites.google.com/site/palm3ddocumentation/home/TwoFiles.png)

In this case, the first file contains 200 images taken at z=3 microns depth, and the second file contains 20 images taken at z=0 microns depth. The first file gives us the PALM data we want, and the second file gives us the information we need to subtract thermal drifts.

After many repetitions of steps 4-7, the acquisition directory might look like this:

![http://sites.google.com/site/palm3ddocumentation/home/ManyFiles.png](http://sites.google.com/site/palm3ddocumentation/home/ManyFiles.png)

The first file, and every subsequent even-numbered file in the acquisition, contains 200 images taken at z=3 microns depth. The second file, and every subsequent odd-numbered file in the acquisition, contains 20 images taken at z=0 microns depth for drift tracking. All this data would go in a single 'acquisition' folder in the experiment directory.

After several acquisitions are taken, the experiment directory might look like this:

![http://sites.google.com/site/palm3ddocumentation/home/JumptrackingData.png](http://sites.google.com/site/palm3ddocumentation/home/JumptrackingData.png)

Each acquisition folder starts with 'z=', and is named in a way that makes it easy to remember how deep in the sample the data was taken. In each acquisition folder is 50 repetitions of the 200 images / 20 tracking images pattern described above. If you follow this pattern, the graphical interface will easily understand the structure of your data.

Of course, if you skip the graphical interface and use 'palm\_3d.py' directly with python, you can structure your data however you want. I don't plan to document this on the wiki anytime soon, but take a look at the code automatically generated by 'palm\_gui.py' for some clues how to proceed with this approach.


## Footnotes ##
  1. palm3d also supports TIFF data, but this format is more complicated, slower to load, and much less tested, so use raw binary unless you want to help us debug our TIFF support.
  1. We've found that our piezoelectric z-positioner can make multi-micron jumps and return to the same position, within better than 50 nanometers if we're careful. 'Careful' means we wait a few tens of milliseconds after each jump for vibrations to ring down, and  we don't wait more than a minute between jumps to minimize thermal drifts.
  1. The image of the stylized microscope is adapted from [here](http://www.flickr.com/photos/neeleshbhandari/3155559217/), which I think is ok since it's [CC licensed](http://creativecommons.org/licenses/by-sa/2.0/deed.en). Thanks, neeleshbhandari!
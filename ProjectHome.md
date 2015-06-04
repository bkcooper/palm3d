# News #
Have you tried out palm3d? We'd love to hear how it went. We need [your feedback](http://www.nibib.nih.gov/Research/Intramural/HighResolutionOpticalImaging/York) to make sure the software works on everyone's data, not just ours.
# Introduction #
Fluorescence microscopes are an awesome tool for biologists, but get blurry if you zoom in too much. It turns out you can remove a lot of this blur if your fluorescent molecules glow one at a time. (See [PALM](http://www.ncbi.nlm.nih.gov/pubmed/16902090), [STORM](http://www.ncbi.nlm.nih.gov/pmc/articles/PMC2700296/), [FPALM](http://www.ncbi.nlm.nih.gov/pmc/articles/PMC1635685/) for examples).

A drawback of this trick is that the raw data from your microscope looks like gibberish:

<a href='http://www.youtube.com/watch?feature=player_embedded&v=ffFDMgxcC_8' target='_blank'><img src='http://img.youtube.com/vi/ffFDMgxcC_8/0.jpg' width='425' height=344 /></a>

You need software to turn this data into a three-dimensional image like this one:

<a href='http://www.youtube.com/watch?feature=player_embedded&v=yEM4gwMavKI' target='_blank'><img src='http://img.youtube.com/vi/yEM4gwMavKI/0.jpg' width='425' height=344 /></a>

That's what 'palm3d' does.

# Getting started #
Start with our step-by-step [tutorial](http://code.google.com/p/palm3d/wiki/Tutorial). After you're used to the code, try processing some of your [own data](http://code.google.com/p/palm3d/wiki/TakingData), and let us know how it goes.

# Why use palm3d? #
Other software exists to do the same thing. For example, [QuickPALM](http://code.google.com/p/quickpalm/) is an excellent project that tackles a similar problem. 'palm3d' tries to be hardware-agnostic and reasonably fast, with a focus on three-dimensional super-resolution imaging.

We developed palm3d while PALM-imaging [thick cells](http://dx.doi.org/10.1038/nmeth.1571) tagged with fluorescent proteins. palm3d overcomes many difficulties presented by these samples:
  * [Water-immersion objectives](http://www.microscopyu.com/articles/optics/waterimmersionobjectives.html) are better than oil objectives for thick samples, but can be [inconsistent](http://code.google.com/p/palm3d/wiki/TakingData#Taking_calibration_data); palm3d deals well with these inconsistencies.
  * Drift tracking is difficult in thick samples, but palm3d corrects drift using a [simple trick](http://code.google.com/p/palm3d/wiki/TakingData#Taking_PALM_data_the_flexible_way).
  * Genetically expressed fluorescent proteins are dim compared to other PALM/STORM tags, and thick samples give noisy images due to autofluorescence. We've tested palm3d in samples up to ~10 microns thick, and it works well with fluorescent proteins.

# What hardware do I need? #
## Short answer ##
A scope that can do 2D PALM/STORM, with a precise axial sample positioner.

## Long answer ##
[Read our paper](http://dx.doi.org/10.1038/nmeth.1571), where we describe how to build a 3D PALM for thick samples. If you build the same scope, it should work for samples thicker than 10 microns, tagged with two-photon activatable fluorophores. We've had the best luck with PA-mCherry, but we suspect there are other good two-photon activatable fluorophores. Let us know what you try!

If you don't have a femtosecond laser, or don't want to use two-photon activatable fluorophores, you can use one-photon activation instead. One-photon activation works great for thin samples (~1 micron or less), and should be ok even for moderately thick samples (~3 microns).

Most existing 2D PALM/STORM scopes should work fine with palm3d, as long as they have a precise axial positioner for [calibration](http://code.google.com/p/palm3d/wiki/TakingData#Taking_calibration_data) and [jump-tracking](http://code.google.com/p/palm3d/wiki/TakingData#So_we_use_'jump-tracking'). We recommend a water-immersion objective to avoid [depth-dependent aberrations](http://www.microscopyu.com/articles/optics/waterimmersionobjectives.html). Our paper describes how we put a cylindrical lens in our microscope's imaging path to increase z-resolution, but this is optional. palm3d works either way, which makes it easy to experiment with different hardware configurations.

If you plan to take a lot of PALM data, consider getting a nice computer for processing. Most of the PALM data for our paper was taken on a single-core workstation with a single internal hard drive, but we were taking data faster than we could process it. We recently upgraded to a multi-core workstation with a small RAID array, and now we process data faster than we acquire it. It's much nicer to see your PALM image as you take it.
=News=
Have you tried out palm3d? We'd love to hear how it went. We need [http://www.nibib.nih.gov/Research/Intramural/HighResolutionOpticalImaging/York your feedback] to make sure the software works on everyone's data, not just ours. 
=Introduction=
Fluorescence microscopes are an awesome tool for biologists, but get blurry if you zoom in too much. It turns out you can remove a lot of this blur if your fluorescent molecules glow one at a time. (See [http://www.ncbi.nlm.nih.gov/pubmed/16902090 PALM], [http://www.ncbi.nlm.nih.gov/pmc/articles/PMC2700296/ STORM], [http://www.ncbi.nlm.nih.gov/pmc/articles/PMC1635685/ FPALM] for examples).

A drawback of this trick is that the raw data from your microscope looks like gibberish:

<wiki:video url="http://www.youtube.com/watch?v=ffFDMgxcC_8"/>

You need software to turn this data into a three-dimensional image like this one:

<wiki:video url="http://www.youtube.com/watch?v=yEM4gwMavKI"/>

That's what 'palm3d' does.

=Getting started=
Start with our step-by-step [http://code.google.com/p/palm3d/wiki/Tutorial tutorial]. After you're used to the code, try processing some of your [http://code.google.com/p/palm3d/wiki/TakingData own data], and let us know how it goes.

=Why use palm3d?=
Other software exists to do the same thing. For example, [http://code.google.com/p/quickpalm/ QuickPALM] is an excellent project that tackles a similar problem. 'palm3d' tries to be hardware-agnostic and reasonably fast, with a focus on three-dimensional super-resolution imaging. 

We developed palm3d while PALM-imaging [http://dx.doi.org/10.1038/nmeth.1571 thick cells] tagged with fluorescent proteins. palm3d overcomes many difficulties presented by these samples:
  * [http://www.microscopyu.com/articles/optics/waterimmersionobjectives.html Water-immersion objectives] are better than oil objectives for thick samples, but can be [http://code.google.com/p/palm3d/wiki/TakingData#Taking_calibration_data inconsistent]; palm3d deals well with these inconsistencies.
  * Drift tracking is difficult in thick samples, but palm3d corrects drift using a [http://code.google.com/p/palm3d/wiki/TakingData#Taking_PALM_data_the_flexible_way simple trick].
  * Genetically expressed fluorescent proteins are dim compared to other PALM/STORM tags, and thick samples give noisy images due to autofluorescence. We've tested palm3d in samples up to ~10 microns thick, and it works well with fluorescent proteins.

=What hardware do I need?=
==Short answer==
A scope that can do 2D PALM/STORM, with a precise axial sample positioner.

==Long answer==
[http://dx.doi.org/10.1038/nmeth.1571 Read our paper], where we describe how to build a 3D PALM for thick samples. If you build the same scope, it should work for samples thicker than 10 microns, tagged with two-photon activatable fluorophores. We've had the best luck with PA-mCherry, but we suspect there are other good two-photon activatable fluorophores. Let us know what you try!

If you don't have a femtosecond laser, or don't want to use two-photon activatable fluorophores, you can use one-photon activation instead. One-photon activation works great for thin samples (~1 micron or less), and should be ok even for moderately thick samples (~3 microns).

Most existing 2D PALM/STORM scopes should work fine with palm3d, as long as they have a precise axial positioner for [http://code.google.com/p/palm3d/wiki/TakingData#Taking_calibration_data calibration] and [http://code.google.com/p/palm3d/wiki/TakingData#So_we_use_'jump-tracking' jump-tracking]. We recommend a water-immersion objective to avoid [http://www.microscopyu.com/articles/optics/waterimmersionobjectives.html depth-dependent aberrations]. Our paper describes how we put a cylindrical lens in our microscope's imaging path to increase z-resolution, but this is optional. palm3d works either way, which makes it easy to experiment with different hardware configurations.

If you plan to take a lot of PALM data, consider getting a nice computer for processing. Most of the PALM data for our paper was taken on a single-core workstation with a single internal hard drive, but we were taking data faster than we could process it. We recently upgraded to a multi-core workstation with a small RAID array, and now we process data faster than we acquire it. It's much nicer to see your PALM image as you take it.

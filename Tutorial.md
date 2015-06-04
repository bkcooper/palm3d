

# Introduction #
A real 3D PALM dataset of a thick sample is huge, often tens or even hundreds of gigabytes, and much too big to make a good example. So, I made some artificial raw PALM data, based on the [bat cochlea example](http://rsb.info.nih.gov/ij/images/bat-cochlea-volume.zip) in [ImageJ](http://rsbweb.nih.gov/ij/); this way you can try out palm3d before you go to the trouble of building a 3D superresolution microscope.

# Finding your data #
After you've [installed](http://code.google.com/p/palm3d/wiki/Installation) the most recent version of palm3d, download and unzip [tutorial\_data\_vX](http://code.google.com/p/palm3d/downloads/list) (X is the version number). Your palm folder might look like this [(1)](http://code.google.com/p/palm3d/wiki/Tutorial#(1)):

![http://sites.google.com/site/palm3ddocumentation/home/tutorial_start.jpg](http://sites.google.com/site/palm3ddocumentation/home/tutorial_start.jpg)

Inside the 'data' folder is some artificial raw PALM data. The raw data looks like this [(2)](http://code.google.com/p/palm3d/wiki/Tutorial#(2)):

<a href='http://www.youtube.com/watch?feature=player_embedded&v=GIMQMXXhmKo' target='_blank'><img src='http://img.youtube.com/vi/GIMQMXXhmKo/0.jpg' width='425' height=344 /></a>

Open the palm3d\_Windows folder, and double-click on 'palm3d.wsf'. There may be a long pause the first time you do this [(3)](http://code.google.com/p/palm3d/wiki/Tutorial#(3)). Once the 'Browse' dialog opens:

http://sites.google.com/site/palm3ddocumentation/home/tutorial_browse.JPG

...select the 'data' folder, press 'Ok'. You should see this screen:

http://sites.google.com/site/palm3ddocumentation/home/tutorial_nodata.JPG

palm3d is confused. In each experiment folder, it expects to find a calibration folder (which it did, called 'calibration'), and some data folders (which it didn't). By default, palm3d assumes data folders start with the prefix 'z=', but as you can see, our data folders don't:

http://sites.google.com/site/palm3ddocumentation/home/tutorial_datafolder.JPG

Our data folders all start with 'p='. So, we change 'Image folder prefix:' to 'p=', and hit the 'Refresh' button:

http://sites.google.com/site/palm3ddocumentation/home/tutorial_founddata.JPG

All our ducks are in a row, and we're ready to load calibration data.

# Loading calibration data #
## Entering calibration metadata ##
Click the 'Load' button next to 'Calibration folder:'. You'll be asked a series of questions about how the [calibration data is structured](http://code.google.com/p/palm3d/wiki/TakingData#Taking_calibration_data). The tutorial calibration data was taken with one image per position, one repetition, with image dimensions 20 pixels in the up-down direction, and 13 pixels in the left-right direction, so you would answer:

https://sites.google.com/site/palm3ddocumentation/home/calreps.JPG
https://sites.google.com/site/palm3ddocumentation/home/calimages.JPG
https://sites.google.com/site/palm3ddocumentation/home/cal_xpix.JPG
https://sites.google.com/site/palm3ddocumentation/home/cal_ypix.JPG

Note that input to palm3d uses the MATLAB convention: 'x\_pixels' means the up-down direction, and 'y\_pixels' means the left-right direction. This is NOT the same convention ImageJ uses, so be careful. I should probably remove all reference to 'x' and 'y' anyhow, and stick strictly to left-right/up-down.

Anyhow, some extra windows should open, showing your calibration data:

http://sites.google.com/site/palm3ddocumentation/home/cal_slices.JPG?height=277&width=400

Inspect these windows carefully. Does the calibration data look how you expect? Are the calibration files listed in the order you expected? High-quality calibration data is crucial for getting good PALM results, so spend some time looking over these slices.

## Cropping ##

Real calibration data usually needs to be cropped, but the tutorial data is already nicely shaped, so this step doesn't matter too much. The slice labeled '0' seems to have the widest image, so we'll use it: select the black window, type the number 0 and press the Return key. You'll be prompted to click twice in the figure window to crop your calibration stack. Since we don't really need to crop, click the uppermost, leftmost pixel, then click the lowermost, rightmost pixel:

http://sites.google.com/site/palm3ddocumentation/home/cal_cropping.JPG?height=339&width=400

## Inspection ##

If the cropping looks good, return to the black window and press Return (square brackets mean 'y' is the default answer). Press Return again to inspect the cropped, processed calibration stack. You'll see three more figure windows, and three PNG files are saved in the experiment directory:

http://sites.google.com/site/palm3ddocumentation/home/calpngs.JPG

The first two PNGs show you the cropped, processed calibration stack. The purpose of the third (calXC.png) is a little less obvious:

http://sites.google.com/site/palm3ddocumentation/home/calXC.png?height=301&width=400

Three-dimensional PALM is a guessing game. Based on the shape of a molecule's image, palm3d guesses where the molecule is in 'z'. If two different 'z' positions look very similar, it's very hard to make an accurate guess. 'calXC.png' shows the maximum cross-correlation between each slice of the calibration stack. Adjacent slices are similar, and have large cross-correlation; distant slices are not, and don't. When you take your own calibration data, you must watch out for similar but distant slices, which will confuse palm3d and cause poor 'z' localization.

Anyhow, with our calibration data loaded, we're ready to process some PALM data [(4)](http://code.google.com/p/palm3d/wiki/Tutorial#(4)). Select the black window and press Return to close the window.

# Processing the first dataset #

## Entering metadata ##

Before we can start processing the first acquisition, we have to tell palm3d some information about how the data was taken. The first acquisition in the example data is structured [the simple way](http://code.google.com/p/palm3d/wiki/TakingData#Taking_PALM_data_the_simple_way), so this is pretty straightforward. By definition, the axial positioner is at z = 0 if you take data the simple way. To get the rest of the information we need, open the file 'data`_`000.txt' in the folder 'p=00'. Click 'Set metadata' next to this folder's name in the '3D palm data processing' window, and answer the following questions:

http://sites.google.com/site/palm3ddocumentation/home/z_position.JPG
http://sites.google.com/site/palm3ddocumentation/home/datareps.JPG
http://sites.google.com/site/palm3ddocumentation/home/imsperfile.JPG
http://sites.google.com/site/palm3ddocumentation/home/trackingimsperfile.JPG
http://sites.google.com/site/palm3ddocumentation/home/xpix.JPG
http://sites.google.com/site/palm3ddocumentation/home/ypix.JPG

As before, remember that 'x\_pixels' means the up-down direction, like in MATLAB.

The palm3d main screen should now look like this:

http://sites.google.com/site/palm3ddocumentation/home/ready2process.JPG
## Candidate selection ##
Click the 'Process' button. A black window will pop up; select this window, and hit Return (as always, the command in square brackets is the default). You should see something like this:

http://sites.google.com/site/palm3ddocumentation/home/testparameters.JPG?height=306&width=400

The first task is to run through the raw data and pick out candidate particles that might be photoactivated molecules or fiducials. palm3d does this by looking for bright patches or patches that change brightness rapidly. palm3d tries to make intelligent guesses, but it helps a lot for a human to review some raw data and tune detection parameters.

The top panel, labeled 'Unfiltered image', shows the first raw data image. There's a single, obvious bright spot, but real data is not always this nice. Real data often has hot pixels and large background variation. [Bandpass filtering](http://books.google.com/books?id=8uGOnjRGEzoC&lpg=PA337&ots=8tZjTu9__d&dq=%22bandpass%20filter%22%20%22image%20processing%22&pg=PA336#v=onepage&q&f=false) helps ignore these distortions by looking for bright spots that are not too big and not too small:

http://sites.google.com/site/palm3ddocumentation/home/bandpass.JPG

All points brighter than four standard deviations above the mean of the filtered image are marked red. The green box shows where palm3d found a bright particle. Note the units of the colormap are renormalized, divided by the standard deviation of the filtered image.

Select the black window and press Return to move to the next image. Now the screen should look like this:

http://sites.google.com/site/palm3ddocumentation/home/differential_image.JPG?height=248&width=400

There are now four images in the figure window. The left two images are still a raw data frame and a filtered image. A second glowing particle has blinked on, and palm3d caught it:

http://sites.google.com/site/palm3ddocumentation/home/second_particle.JPG

The two images on the right are new. On the upper right is a 'differential image':

http://sites.google.com/site/palm3ddocumentation/home/differential_image_ur.JPG

A differential image shows the difference between the previous image and  the current image. This is useful for catching photoactivated particles as they blink on or bleach off; notice that only the new particle shows up in the differential image. On the lower right is the filtered differential image:

http://sites.google.com/site/palm3ddocumentation/home/differential_image_lr.JPG

Like the filtered image on the left, all pixels in the filtered differential image which are four standard deviations brighter than the mean of the filtered differential image are highlighted in red. palm3d detected the new particle's birth, and draws a green box around it.

Select the black window and press Return to move to the next image:

http://sites.google.com/site/palm3ddocumentation/home/differential_image_2.JPG?height=327&width=400

A third particle is born in this frame; it's pretty dim. palm3d noticed it in the differential image, but not in the regular image. This is bad!

We need to tune detection parameters. Select the black window, type the letter 'n', and press Return. You'll be prompted to input two (somewhat cryptic) values, 'numSTD' and 'numSTD`_`changed'. 'numSTD' tells palm3d how many standard deviations above the mean a pixel must be for detection in the filtered image. 'numSTD`_`changed' is a similar detection threshold for the filtered differential image. Lower 'numSTD' until palm3d notices the third particle:

http://sites.google.com/site/palm3ddocumentation/home/tuning.JPG

numSTD = 1.5, numSTD`_`changed = 4 seems to work:

http://sites.google.com/site/palm3ddocumentation/home/tuning2.JPG?height=333&width=400

Select the black window, and press Return several more times to inspect more raw data frames. This is a good habit to get into; never trust palm3d so much that you don't look over the raw data a little bit yourself.

After you're sick of inspecting raw data, select the black window, type the letter 'j', press Return, type the number '4', and press Return:

http://sites.google.com/site/palm3ddocumentation/home/jump.JPG

This jumps you back to image 4. Notice the purple box in the filtered differential image:

http://sites.google.com/site/palm3ddocumentation/home/death.JPG

Black regions in the filtered differential image are regions that got darker. The purple box indicates palm3d thinks a molecule bleached out, and marks this as a 'death'. The threshold for detecting deaths is 'num\_STD`_`changed', the same as for detecting births. Notice that palm3d missed another death in this frame, though, on the left hand side.

Spend some time trying to adjust 'numSTD`_`changed' to catch this death. Hard, isn't it? If you set 'numSTD`_`changed' low enough (1.5, perhaps), you start picking up spurious molecules elsewhere (false positives). If you don't set it this low (say, 2), you miss the molecule (a false negative). What should you choose?

False negatives waste molecules. False positives waste computation time, and you should be careful to filter them out later. There's always a tradeoff, so after you settle on values for 'numSTD' and 'numSTD`_`changed', spend some time looking at different frames and make sure you like your choice.

For now, set the parameters to numSTD=1.5 and numSTD`_`changed=2.5. Type the letter 'f' and press Return. The figure window should close, and you should see a status message like this:

http://sites.google.com/site/palm3ddocumentation/home/progress.JPG

palm3d is hard at work processing. Now is a good time to get some coffee [(5)](http://code.google.com/p/palm3d/wiki/Tutorial#(5)).

## Processing ##
While you're drinking your coffee, the sub-window launched by palm3d does four things:

  1. Select candidates
  1. Localize candidates
  1. Link localizations
  1. Localize linked candidates

We already saw candidate selection in action. This is disk-intensive, but not very CPU-intensive. During localization, each candidate particle is compared to the calibration stack to estimate its subpixel position in x, y and z. Localization is typically CPU-limited [(6)](http://code.google.com/p/palm3d/wiki/Tutorial#(6)). Linking is an attempt to find pairs of 'birth' and 'death' localizations that match. This tends to filter out spurious localizations, allows averaging multiple frames to increase signal-to-noise, and hopefully improves localization accuracy. Linking generally rejects some good localizations too, so it's not without cost. Linked candidate images are then averaged and relocalized, after which it's time for user input again.

The black window should display progress through these steps [(7)](http://code.google.com/p/palm3d/wiki/Tutorial#(7)). When it's done, it will look like this:

http://sites.google.com/site/palm3ddocumentation/home/doneprocessing.JPG


## Drift correction ##
[Drift tracking](http://code.google.com/p/palm3d/wiki/TakingData#Drift_tracking_is_crucial) is crucial for high-resolution, long-exposure PALM images. We need to identify a tracking particle so palm3d can correct sample drift. Select the black window, type the letter 'p', and press Return. You'll be warned you're about to plot a lot of points; ignore this for now [(8)](http://code.google.com/p/palm3d/wiki/Tutorial#(8)). Type the letter 'n' and press Return. After a pause, you'll see a figure like this:

http://sites.google.com/site/palm3ddocumentation/home/xyz.JPG?height=327&width=400

This figure shows x, y, z, and correlation strength vs. image number for every localization. The solid red horizontal lines are from our tracking particle [(9)](http://code.google.com/p/palm3d/wiki/Tutorial#(9)). We need to tell palm3d to use this particle; type the letter 'a', press Return, and answer the following questions:

http://sites.google.com/site/palm3ddocumentation/home/filter.JPG

We're trying to specify a set of filters that pick out our tracking particle and exclude everything else. To check the results of the filter, type the letter 'p', press Return, type the letter 'n', and press Return:

http://sites.google.com/site/palm3ddocumentation/home/filter_results.JPG?height=286&width=400

Figure 1 shows the localizations caught by our filter, and Figure 2 shows the ones that weren't.

The filter is stored in the experiment directory:

http://sites.google.com/site/palm3ddocumentation/home/fiducial_filter.JPG

If you want to change the filter later, open it in a text editor (I like [IDLE](http://docs.python.org/library/idle.html), but [Notepad](http://en.wikipedia.org/wiki/Notepad_(software)) works in a pinch):

http://sites.google.com/site/palm3ddocumentation/home/idle_vs_notepad.JPG?height=339&width=400

Because the tracking particle isn't on top of many other localizations, our filter can be fairly sloppy. Type the letter 'f' to finish adding filters and press Return twice. Decline the offer to decimate, and you should see these windows:

http://sites.google.com/site/palm3ddocumentation/home/resmooth_me.JPG?height=286&width=400

The blue line shows palm3d's estimate of sample drift, based on the localizations that pass our filter. The user's job is to estimate if the blue line is a good fit. Since the fake data isn't very noisy, our tracking is very good. The estimate is a little wiggly, though, and might benefit from more smoothing.

Type the letter 'r', press Return, type the number '100', and press Return. The resmoothed line should look better, and we're done drift correcting. Press Return, type the number '0', and press Return twice to finish.

## Histogram construction ##
Click the Refresh button in the main palm3d window (or press Control-r):

http://sites.google.com/site/palm3ddocumentation/home/ready_2_bin.JPG

The first acquisition is finished processing, and we're ready to make a 3D image. Click the 'Construct Histogram' button. We'll get a series of questions:

http://sites.google.com/site/palm3ddocumentation/home/min_correlation.JPG
http://sites.google.com/site/palm3ddocumentation/home/min_z_calibration.JPG
http://sites.google.com/site/palm3ddocumentation/home/max_z_calibration.JPG
http://sites.google.com/site/palm3ddocumentation/home/nm_x.JPG
http://sites.google.com/site/palm3ddocumentation/home/nm_y.JPG
http://sites.google.com/site/palm3ddocumentation/home/nm_z.JPG
http://sites.google.com/site/palm3ddocumentation/home/nm_per_bin.JPG
http://sites.google.com/site/palm3ddocumentation/home/min_x.JPG
http://sites.google.com/site/palm3ddocumentation/home/max_x.JPG
http://sites.google.com/site/palm3ddocumentation/home/min_y.JPG
http://sites.google.com/site/palm3ddocumentation/home/max_y.JPG
http://sites.google.com/site/palm3ddocumentation/home/min_z.JPG
http://sites.google.com/site/palm3ddocumentation/home/max_z.JPG
http://sites.google.com/site/palm3ddocumentation/home/linked_yn.JPG

  * Minimum correlation sets a filter to exclude low-quality localizations from the histogram. Our microscope gives good results with 0.4, but you should experiment with this number for best results.
  * Minimum/maximum calibration slice: We've found it's best to exclude localizations that match highly out-of-focus slices of the calibration. The example calibration has 40 slices, so we'll use slices 5 through 35.
  * Nanometers per x/y pixel is determined by your microscope's magnification and your camera's pixel size. The example data uses 100 nm pixels.
  * Nanometers per z pixel is determined by the step size you used when taking calibration data. The example data uses 100 nm z-pixels.
  * Nanometers per histogram bin is a fairly important choice. Too small, and you'll make a scatterplot, and probably overflow memory. Too large, and you limit your resolution artificially. It's best to experiment with this parameter to get best results. For now, we'll use 30.
  * Minimum/maximum x/y is the transverse extent of the histogram, in pixels. The default is to use the same size as the original image. It can be useful to crop the histogram to avoid memory limitations, though.
  * Minimum/maximum z is the axial extent of the histogram, in pixels. The default is the extent of the calibration stack.
  * Finally, we must choose linked or unlinked input. Linking is a tradeoff. It usually decreases the number of localizations, but is resistant to certain types of noise and might give higher precision localizations. For now let's use unlinked input (answer 'no').

All these different numbers may be confusing, at first. It can help to keep in mind we're dealing with three different coordinate systems to describe the positions of our particles:

  1. Physical coordinates - the actual position of the glowing particles.
  1. Pixel coordinates - palm3d uses pixel coordinates everywhere, until histogram creation.
  1. Histogram bin coordinates - user chosen.

Whenever you enter data into palm3d, be aware of which units you're using.

Anyhow, you should see this window, warning you about memory use:

http://sites.google.com/site/palm3ddocumentation/home/memory_warning.JPG

40 MB isn't too bad [(10)](http://code.google.com/p/palm3d/wiki/Tutorial#(10)), so press Return twice to continue. You should see a crude view of a 3D image, from three different directions:

http://sites.google.com/site/palm3ddocumentation/home/3d_views.JPG?height=400&width=387

Our final step is to output the histogram for processing and viewing in other programs. Type the letter 'o' and press return twice to output our raw data. Finally, type the letter 'd' and press return to exit the black window.

# Playing with the histogram in ImageJ #
I'm sure there's a ton of good software for viewing and manipulating three-dimensional data. For now, we'll use [ImageJ](http://rsbweb.nih.gov/ij/).

Import the histogram stack in ImageJ:

http://sites.google.com/site/palm3ddocumentation/home/import_raw.JPG?height=371&width=400
http://sites.google.com/site/palm3ddocumentation/home/find_the_histogram.JPG?height=308&width=400

Since raw data doesn't have headers or footers, we have to tell ImageJ the image dimensions. The file 'histogram.txt' was created along with 'histogram.dat' to record this information:

http://sites.google.com/site/palm3ddocumentation/home/histogram_info.JPG?height=344&width=400

Z-projection is a good way to quickly check if the image is garbage:

http://sites.google.com/site/palm3ddocumentation/home/z_projection.JPG?height=400&width=347

The tracking particle produces a ton of counts in the histogram, so you have to change the brightness and contrast to see the structure:

http://sites.google.com/site/palm3ddocumentation/home/max_projection.JPG

If we convert the data to 8-bit (Image->Type->8-bit), we can view it in 3D (Image->Stacks->3D Project...):

<a href='http://www.youtube.com/watch?feature=player_embedded&v=O8C7Z5JsO24' target='_blank'><img src='http://img.youtube.com/vi/O8C7Z5JsO24/0.jpg' width='425' height=344 /></a>

The data is grainy, like any unsmoothed PALM data. More annoying, the data seems to be cut off! This is because the sample is thicker than our calibration stack.

Most fluorescent proteins are only visible if they're nearly in-focus. A truncated image is the best we can do without taking data at [multiple depths in the sample](http://code.google.com/p/palm3d/wiki/TakingData#...but_drift_tracking_is_tricky_in_thick_samples).

# Processing the second and third datasets #

## Entering metadata for jump-tracking ##
The second acquisition in the sample data (p=n15) is structured the [flexible way](http://code.google.com/p/palm3d/wiki/TakingData#Taking_PALM_data_the_flexible_way):

http://sites.google.com/site/palm3ddocumentation/home/flexible_way.JPG

This lets us image thicker samples without having to use multiple tracking particles. The files alternate: 200 images taken deeper in the sample, followed by 20 images taken with the tracking particle in focus [(11)](http://code.google.com/p/palm3d/wiki/Tutorial#(11)). This pattern repeats 10 times, so when we set metadata for this acquisition, we would answer:

https://sites.google.com/site/palm3ddocumentation/home/n15_z.JPG
https://sites.google.com/site/palm3ddocumentation/home/n15_reps.JPG
https://sites.google.com/site/palm3ddocumentation/home/n15_ims_per_file.JPG
https://sites.google.com/site/palm3ddocumentation/home/n15_tr_ims_per_file.JPG

'x\_pixels' and 'y\_pixels' are the same as before, so I didn't bother showing them. We entered '-15' for the slice z-position. This is the distance, in z-pixels, that we [move the sample axially](http://code.google.com/p/palm3d/wiki/TakingData#So_we_use_'jump-tracking') away from the tracking particle to take each set of 200 images. By definition, the sample is at z=0 when the tracking particle is in-focus. Be careful about your sign here! The positive z-direction is defined by the [calibration data](http://code.google.com/p/palm3d/wiki/TakingData#Taking_calibration_data). Your screen should look like this:

http://sites.google.com/site/palm3ddocumentation/home/n15_process.JPG

Click the 'Process' button, just like before. The familiar black window should pop up; press Return to test selection parameters. Keep pressing Return to search through some data frames:

http://sites.google.com/site/palm3ddocumentation/home/n15_parameters.JPG?height=251&width=400

It seems like numSTD=4 and numSTD`_`changed=4 are doing a pretty good job identifying candidates.

Press 'f' to begin processing:

http://sites.google.com/site/palm3ddocumentation/home/n15_processing.JPG

## Processing in parallel ##
While the p=n15 acquisition is processing, we might as well process the p=n30 acquisition in parallel. Minimize the black window while it processes, and return to the '3D palm data processing' window.

Click the 'Set metadata' button in the 'p=n30' acquisition row. This data was taken at a different depth in the sample:

http://sites.google.com/site/palm3ddocumentation/home/n30_z.JPG

But the rest of your answers ('repetitions', 'slice\_images\_per\_file', 'tracking\_images\_per\_file', 'x\_pixels', 'y\_pixels') are the same as for p=n15. Click the 'Process' button in the 'p=n30' row.

A new black window will pop up. Press Return to begin testing selection parameters again:

http://sites.google.com/site/palm3ddocumentation/home/n30_candidates.JPG?height=258&width=400

This last acquisition is quite sparse. Many of the frames contain no molecules, and the tracking particle is completely invisible in the first 200 frames.

Again, numSTD=4, numSTD`_`changed=4 seems sufficient to catch the particles that appear. After searching through enough frames to verify this, type 'f' and press Return to start processing the second set.

## Drift correction, again ##
When the 'p=n15' dataset is finished processing, select its black window [(12)](http://code.google.com/p/palm3d/wiki/Tutorial#(12)) and do drift correction just like before:

http://sites.google.com/site/palm3ddocumentation/home/n15_drift.JPG

The 'piezoMax' and 'piezoMin' filters restrict us to localizations where the tracking particle is in focus. Because the tracking data is sparse, we need more smoothing than before:

http://sites.google.com/site/palm3ddocumentation/home/n15_resmooth.JPG?height=275&width=400

A smoothing sigma of 200 seems to give reasonable results. After setting the smoothing sigma, press Return, type the number '0', and press Enter twice to finish drift correcting this acquisition.

Now switch to the black window that was processing the 'p=n30' acquisition. For the sake of education, let's make some mistakes while drift correcting:

http://sites.google.com/site/palm3ddocumentation/home/n30_drift_mistakes.JPG

Enter these bad choices, and press Return to determine sample drift.

This does not result in a good measurement of drift:

http://sites.google.com/site/palm3ddocumentation/home/n30_bad_drift.JPG?height=253&width=400

The variability is 10's of pixels; we can't use this to produce a high-quality PALM image.

To fix our filter, select the black window, type the letter 'e', and press Return. You'll be prompted to edit 'n30\_palm\_selected\_fiducials.py'. It should be in the experiment folder:

http://sites.google.com/site/palm3ddocumentation/home/n30_edit_this.JPG

If you know how, open this file in [IDLE](http://en.wikipedia.org/wiki/IDLE_(Python)). If not, open it in [Notepad](http://en.wikipedia.org/wiki/Notepad_(software)):

http://sites.google.com/site/palm3ddocumentation/home/n30_editing.JPG

Change the filter values to something more sensible like we used before:

http://sites.google.com/site/palm3ddocumentation/home/n30_edited.JPG

When you're done editing the file, type 'Control-s' to save, (or File->Save with the mouse).

Select the black window and press Return twice to see the results:

http://sites.google.com/site/palm3ddocumentation/home/n30_good_drift.JPG?height=251&width=400

This is much better. Select the black window, press Return, type the number '0', and press Return twice to finish drift correction.

## Combining the histograms ##
Refresh the '3D palm data processing' window, and you should see this:

http://sites.google.com/site/palm3ddocumentation/home/almost_finished.JPG

Cross your fingers and click the 'Construct histogram' button. You'll be asked if you want to use old parameters. These parameters are stored in the experiment folder:

http://sites.google.com/site/palm3ddocumentation/home/histogram_metadata.JPG?height=400&width=330

You can read the parameters in Notepad:

http://sites.google.com/site/palm3ddocumentation/home/histogram_metadata_2.JPG

These parameters are almost good, but will cut off the sample in the z-direction. Change 'maximum\_z':

http://sites.google.com/site/palm3ddocumentation/home/histogram_metadata_3.JPG

Then you can click 'Yes' in the 'Use old parameters' dialog. Click 'no' next, to use unlinked input. A black window will pop up, and you'll get the same warning about memory, ~60 MB this time. Press Return to continue, and Return again to display the 3D histogram. Type the letter 'o' and hit Return twice to output the new histogram. Finally, type the letter 'd' and hit Return to finish histogram generation.

Just like before, the information about this histogram is stored in 'histogram.txt'. In ImageJ, the 3D projection looks like this:

<a href='http://www.youtube.com/watch?feature=player_embedded&v=NABTe5EoTQc' target='_blank'><img src='http://img.youtube.com/vi/NABTe5EoTQc/0.jpg' width='425' height=344 /></a>

# Ok, what now? #
Try it out on some of your data!

I'm especially interested to see how palm3d works with other people's hardware. I'd also like to see how it works with other fluorescent tags like Cy-dyes, caged rhodamine, or really anything other than the photoactivatible fluorescent proteins we've been using.

Make sure to structure your data in a way [palm3d will understand](http://code.google.com/p/palm3d/wiki/TakingData). If you have some unusual data structure, consider learning the programmatic interface to palm3d instead of the graphical interface. The .py files produced by the graphical interface should be a clue where to start. I tried to make the source code readable too, but I know how fun it isn't to read someone else's code.

Good luck, and please let me know how it goes!

# Footnotes #
### (1) ###
This tutorial assumes you're using the stand-alone, no-install Windows version of palm3d. If you're using another operating system or have installed python yourself, you're probably qualified to figure out how to make this work on your system.
### (2) ###
You can view the raw data in ImageJ using File->Import->Raw...; The raw data has a '.dat' extension, and the metadata you need is stored in a similarly named .txt file. For example, the file 'data\_000.dat' in the 'data/p=00' folder has metadata stored in 'data\_000.txt' in the same folder.
### (3) ###
I think this pause is python compiling files? It doesn't happen on my Ubuntu machine. It does happen on our NIH laptops, but they're all crippled by antivirus and disk encryption.
### (4) ###
Unless, of course, something went wrong in the calibration loading step. If so, open the 'calibration' folder in the experiment directory, delete the file 'calibration.pkl', and start again from ['Loading calibration data'](http://code.google.com/p/palm3d/wiki/Tutorial#Loading_calibration_data)
### (5) ###
Since the processing happens in an independent subprocess, you should parallelize this step for real data. When we take a lot of PALM data, we process it on a 16-core machine and usually have a ton of processing windows open at once. This is just a tutorial though, so chill out for now.
### (6) ###
Localization involves a 2D FFT of each candidate image, pointwise multiplication by the 3D calibration stack, a downsampled 3D inverse FFT, and several matrix-multiplication inverse fourier transforms. Each candidate image is the same size as a calibration image, so localization speed depends strongly on the the size of the calibration cropping window; smaller is faster. Typical speeds on a single-core ~GHz CPU are 10-100 localizations per second. Multi-core systems with many parallel localization processes may be disk-limited.
### (7) ###
Each of these steps saves results to disk. If a step completes, you shouldn't have to repeat it. If something goes wrong (like a power outage) in the middle of a processing step, the saved results can become corrupted. If you end up having to delete corrupted results by hand for some reason, the save files are in the acquisition folder, named `*`palm\_candidates, `*`palm\_localizations, `*`palm\_localizations\_linked, and `*`palm\_particles.
### (8) ###
Large PALM acquisitions can have millions of localizations, which would take a long time to plot. The 'decimation' option lets you plot a subset of localizations. For example, to plot every other localization, you would use a decimation factor of '2'. To plot every tenth localization, use '10', etc. Once you've defined a nice filter to pick out your tracking particle and you've moved on to determining sample drift, you shouldn't need to decimate anymore.
### (9) ###
You might also notice a pair of horizontal lines at the extremes of the 'z' plot. Bad localizations tend to cluster here, so we'll be sure to filter these results out.
### (10) ###
I wasn't as careful as I could have been when I coded histogram binning. Someday I'll fix it so a 40 MB histogram uses 40 MB of system memory, but it uses more than that today. Windows XP has trouble with histograms bigger than ~150 MB. 64-bit Ubuntu can do larger histograms, but not 10x larger.
### (11) ###
You can tell how many images are in each '.dat' file by opening the similarly named '.txt' file in the same folder. For example, 'data\_000.txt' in the 'p=n15' folder describes the images in 'data\_000.dat'.
### (12) ###
If you lose track of which black window is which, you can scroll up to the top of the window. It will display the image acquisition folder it's processing.
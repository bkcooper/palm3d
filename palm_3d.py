"""
palm_3d written by Andrew York
Version 1.7.2
3d localization of single fluorescent molecules and nanoparticles
based on experimental calibration.

"""
version={
    'major': 1,
    'minor': 7,
    'revision': 2,
    'build': 5}
import os
from scipy.ndimage import gaussian_laplace

def load_microscope_image(
    imageFile,
    verbose = True
    ):
    """Load a 16-bit TIFF saved by ImageJ, correcting for big-endian
    vs. little-endian problems."""
    import os
    from PIL import Image
    import scipy

    if os.name is 'posix':
        """On my Linux version of PIL, ImageJ uint16's need to be
        byteswapped:"""
        if verbose: print "Correcting endian conversion..."
        fullImage = 1.0*scipy.asarray(
            Image.open(imageFile), dtype = scipy.uint16
            ).byteswap()
    if os.name is 'nt':
        """Ugly workaround, since 16-bit TIFFs don't seem to convert
        correctly between PIL and Numpy on our Windows machine. On the
        other hand, they get big/little endian correct:"""
        pilImage = Image.open(imageFile)
        fullImage = 1.0*scipy.array(
            list(pilImage.getdata()),
            dtype = scipy.uint16
            ).reshape(pilImage.size[1],pilImage.size[0])
    """ImageJ tif files seem to be big-endian 16-bit by default. PIL's
    Image function sometimes interprets them as little-endian, so be
    careful to fix this with .byteswap(). The 1.0* converts to
    float."""
    if verbose:
        print fullImage.shape
        print 'Max value:',fullImage.max()
        print 'Min value:',fullImage.min()
    return fullImage

def load_raw_andor_slice(fileName, whichZ, xyShape='ask'):
    """
    Our Andor software often saves PALM image sequences in a 'raw'
    format, binary 16-bit unsigned integers. Like in ImageJ, if the
    user knows the x, y, shape and type of the data, we can load it as
    a series of slices.

    imageFile is a string giving the the name of the file. whichZ is
    an integer specifying which image to load from the stack. xyShape
    can be a tuple (x, y), or 'ask', to set the shape interactively.
    If whichZ is bigger than the number of images in the file, this
    function fails with an error.

    Note that Andor describes image shapes like this:
    {left,right,bottom,top}:      310,662,645,850

    The correct xyShape would be:
    (right - left + 1, top - bottom + 1)
    
    """
    import scipy

    if xyShape == 'ask':
        print "Number of x-pixels:",
        xPix = int(raw_input())
        print "Number of y-pixels:",
        yPix = int(raw_input())
    else:
        (xPix, yPix) = xyShape
    dataType = scipy.dtype(scipy.uint16) ##Unsigned 16-bit integers for now.
    sliceSize = xPix * yPix * dataType.itemsize
    imageFile = open(fileName, 'rb')
    imageFile.seek(whichZ * sliceSize)
    data = 1.0*scipy.fromfile(
        imageFile, dtype = dataType, count = xPix*yPix).reshape(xPix, yPix)
    imageFile.close()
    return data

def get_sif_info(fileName):
    """The Andor .sif header is not simple, or documented. YOU come up
    with a faster way to pull out the number of images and the image
    pixel shape!"""
    f = open(fileName, 'rb')
    header = f.read(5000) ##Random, seems-to-be-large-enough number of bytes.
    if header[0:35] != 'Andor Technology Multi-Channel File':
        raise UserWarning("Header doesn't start like an Andor .sif")

    """Find the description of the image shape:"""
    firstPixNum = header.find('Pixel number') ## Pixel shape soon after
    shapeData = header[firstPixNum:firstPixNum + 300].splitlines()
    numLine = shapeData[2]
    (startIm, endIm) = numLine.split()[5:7]
    numImages = 1 + int(endIm) - int(startIm)
    coordLine = shapeData[3]
    (left, top, right, bottom) = coordLine.split()[1:5] ##Check this!
    (xPix, yPix) = (1 + int(right) - int(left), 1 + int(top) - int(bottom))
    """Determine the offset of the binary data:"""
    offset = firstPixNum + sum([1 + len(i) for i in shapeData[0:4]])
    f.seek(offset)
    numLines = 0
    numBytes = 0
    while numLines < numImages:
        b = f.read(1)
        numBytes += 1
        if b == '\n':
            numLines += 1
    offset += numBytes
    f.close()
    return (xPix, yPix, numImages, offset)

def load_sif_image(fileName, imageNum, xPix, yPix, offset):
    """After you've read the .sif image info from the header, you can
    load an arbitrary image pretty quickly. imageNum starts at 0"""
    import numpy

    f = open(fileName, 'rb')
    imageSize = 4 * xPix * yPix
    f.seek(offset + imageSize * imageNum)
    image = numpy.fromfile(
        f, count=xPix*yPix, dtype=numpy.float32).reshape(xPix, yPix)
    f.close()
    return image

def _im_load(self, imageName):
    if self.im_format == 'tif':
        return load_microscope_image(
            os.path.join(self.imFolder, imageName), verbose=False)
    if self.im_format == 'raw':
        (fileName, whichZ) = imageName.split('*')
        return load_raw_andor_slice(
            fileName=os.path.join(self.imFolder, fileName),
            whichZ=int(whichZ), xyShape=self.im_xy_shape)
    if self.im_format == 'sif':
        (fileName, whichZ) = imageName.split('*')
        (xPix, yPix, numImages, offset) = self.sifInfo.setdefault(
            fileName, get_sif_info(fileName))
        return load_sif_image(
            os.path.join(self.imFolder, fileName),
            imageNum=whichZ, xPix=xPix, yPix=yPix, offset=offset)
    if self.im_format == 'custom':
        return self.custom_im_load(imageName)
    raise UserWarning("Unrecognized image data format")
    return None

def _cal_load(self, imageName):
    if self.cal_format == 'tif':
        return load_microscope_image(
            os.path.join(self.calFolder, imageName), verbose=False)
    if self.cal_format == 'raw':
        (fileName, whichZ) = imageName.split('*')
        return load_raw_andor_slice(
            fileName=os.path.join(self.calFolder, fileName),
            whichZ=int(whichZ), xyShape=self.cal_xy_shape)
    if self.cal_format == 'sif':
        (fileName, whichZ) = imageName.split('*')
        (xPix, yPix, numImages) = self.sifInfo.setdefault(
            fileName + '*calibration', get_sif_info(fileName))
        return load_sif_image(
            os.path.join(self.calFolder, fileName),
            imageNum=whichZ, xPix=xPix, yPix=yPix, numImages=numImages)
    if self.cal_format == 'custom':
        return self.custom_cal_load(imageName)
    raise UserWarning("Unrecognized calibration data format")
    return None

def _get_xy_slices(self):
    import warnings #To suppress ginput deprecation warning
    import scipy, pylab
    print 'Calibration stack is not cropped yet. Plotting...'
    xyShape = self.cal_load(self.calImages[0]).shape
    print "Calibration image XY shape:", xyShape
    stepSize = max(len(self.calImages)//20, 1)
    imagesSlice = slice(0, len(self.calImages), stepSize)
    numFrames = min(len(self.calImages[imagesSlice]), 20)
    calibration = scipy.zeros((
        xyShape[0], xyShape[1], numFrames))
    for c in range(numFrames):
        calibration[:,:,c] = self.cal_load(self.calImages[c*stepSize])
    plot_slices(calibration, labelList=range(len(self.calImages))[imagesSlice])
    while True:
        print 'Cropping frame:',
        cropping_frame = raw_input()
        try:
            cropping_frame = int(cropping_frame)
            break
        except ValueError:
            print "Type an integer, hit return."
    calibration=self.cal_load(self.calImages[cropping_frame])
    pylab.clf()
    pylab.imshow(calibration)
    pylab.title('Click upper left and lower right to crop')
    pylab.gcf().show()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        print "Click on the upper left cropping pixel..."
        (y1, x1) = pylab.ginput(timeout = 0)[0]
        print "Click on the lower right cropping pixel..."
        (y2, x2) = pylab.ginput(timeout = 0)[0]
    xSlice = slice(int(x1),int(x2))
    ySlice = slice(int(y1),int(y2))
    pylab.clf()
    pylab.imshow(calibration[xSlice, ySlice], interpolation='nearest')
    print "Cropping ok? [y]/n:",
    croppingOk = raw_input()
    if croppingOk == 'n':
        pylab.close('all')
        return self.get_xy_slices()
    else:
        return xSlice, ySlice

def _smoothing_window(calibrationShape):
    import scipy
    from scipy.fftpack import ifftshift
    from scipy.signal import get_window

    (x, y, z) = calibrationShape
    (loPass, hiPass) = (('gaussian', 7), ('gaussian', 1))
    Wx_lo = (ifftshift(get_window(loPass, x)).reshape(x, 1))
    Wy_lo = (ifftshift(get_window(loPass, y)).reshape(1, y))
    Wx_hi = (ifftshift(get_window(hiPass, x)).reshape(x, 1))
    Wy_hi = (ifftshift(get_window(hiPass, y)).reshape(1, y))
    W = (Wx_lo*Wy_lo * (1 - Wx_hi*Wy_hi))
    return W
    
def _detection_filter(imageIn):
    return -1 * gaussian_laplace(imageIn, sigma=4)

def _image_to_molecule_locations(
    imageIn,
    image_num,
    xyShape,
    previousFrame=None,
    unfilteredImage=None,
    unfilteredPreviousFrame=None,
    numSTD=4,
    numSTD_changed=None,
    showResults=False,
    ):
    """
    Given a 2d array 'imageIn', finds the bright (or changed) regions
    of the image that we think are single particles.

    'xyShape' is a 2-tuple giving the XY shape of the calibration
    stack used to localize particles in the image. 'numSTD' is the
    number of standard deviations of brightness a particle must rise
    above the image mean to be considered a particle. 'showSteps'
    displays some of the particle selection process, and can be useful
    for debugging and choosing other parameters.

    """
    import scipy

    """Groundwork:"""
    calibrationShape = scipy.asarray(xyShape)
    imageIn = scipy.asarray(imageIn)
    if numSTD_changed is None:
        numSTD_changed = numSTD
    if previousFrame is None:
        imDiff = scipy.zeros(imageIn.shape)
    else:
        imDiff = imageIn - previousFrame
    if showResults:
        if unfilteredImage is None:
            unfilteredImage = scipy.zeros(imageIn.shape)
            unfilteredDiff = scipy.zeros(imageIn.shape)
        else:
            if unfilteredPreviousFrame is None:
                unfilteredDiff = scipy.zeros(imageIn.shape)
            else:
                unfilteredDiff = unfilteredImage - unfilteredPreviousFrame
    locs, avg, std, thresh = [], [], [], []
    for (im, num) in (
        (imageIn, numSTD),
        (imDiff, numSTD_changed),
        (-imDiff, numSTD_changed)):
        avg.append(im.mean()) #Image average
        std.append(im.std()) #Image standard deviation
        thresh.append(im > (avg[-1] + num*std[-1])) ##Brightness threshold
        locs.append([])
        """Zero the image, except for the bright or changed spots:"""
        peaksImage = im * thresh[-1]
        """Chop out the bright spots one at a time:"""
        while peaksImage.max() > 0:
            peakLocation = scipy.unravel_index(
                peaksImage.argmax(),
                peaksImage.shape)
            """The lower left corner of the chop region:"""
            LLcorner = scipy.asarray(peakLocation - calibrationShape//2)
            appendMolecule = True
            """Don't include molecules that touch the image edge:"""
            if (LLcorner <= 0).any():
                appendMolecule = False
                LLcorner = LLcorner * (LLcorner >= 0)
            if ((LLcorner + calibrationShape) >= imageIn.shape).any():
                appendMolecule = False
            molX = slice(LLcorner[0],(LLcorner[0] + calibrationShape[0]))
            molY = slice(LLcorner[1],(LLcorner[1] + calibrationShape[1]))
            if appendMolecule:
                locs[-1].append((molX, molY))
            """Zero out the chopped region:"""
            peaksImage[molX, molY] = 0
    if showResults:
        import pylab
        pylab.suptitle('Image %i. numSTD: %.2f, %.2f'%(
            image_num, numSTD, numSTD_changed))
        for i in range(2):
            if unfilteredPreviousFrame is None and i > 0:
                continue
            pylab.subplot(2, 2, i+1)
            pylab.imshow(
                (unfilteredImage, unfilteredDiff)[i],
                interpolation='nearest', cmap=pylab.cm.gray)
            pylab.colorbar(pad=0,shrink=0.6)
            pylab.title(("Unfiltered image", "Differential image")[i])
        for i in range(2):
            if previousFrame is None and i > 0:
                continue
            (normIm, taggedIm) = _spots_and_slices(
                (imageIn, imDiff)[i], avg[i], std[i],
                thresh[i], locs[i], ([],locs[2])[i])
            pylab.subplot(2, 2, i+3)
            pylab.imshow(normIm, interpolation='nearest', cmap=pylab.cm.gray)
            pylab.colorbar(pad=0, shrink=0.6)
            pylab.imshow(taggedIm, interpolation='nearest')
            pylab.xlabel(
                'Filtered' + ('',' differential')[i] +
                ' image.\nAvg:%.1f SD:%.2f num:%i'%(
                    avg[i], std[i], len(locs[i])))

    brights = locs[0]
    births = locs[1]
    deaths = locs[2]
    return (brights, births, deaths)

def _spots_and_slices(im, av, sd, thresh, locs, darklocs):
    """Utility function for _image_to_molecule_locations()"""
    import scipy
    normIm = (im - av) * 1.0 / sd 
    taggedIm = scipy.zeros((im.shape[0],im.shape[1],3))
    taggedIm[:,:,0] = (im - im.min()) * 1.0 /(im.max() - im.min())
    taggedIm[:,:,1] = taggedIm[:,:,0] * (1 - thresh)
    taggedIm[:,:,2] = taggedIm[:,:,0] * (1 - thresh)
    for (molX, molY) in locs:
        taggedIm[molX.start, molY, 1:2] = 1
        taggedIm[molX.stop, molY, 1:2] = 1
        taggedIm[molX, molY.start, 1:2] = 1
        taggedIm[molX, molY.stop, 1:2] = 1
    for (molX, molY) in darklocs:
        taggedIm[molX.start, molY, 1:2] = 0
        taggedIm[molX.stop, molY, 1:2] = 0
        taggedIm[molX, molY.start, 1:2] = 0
        taggedIm[molX, molY.stop, 1:2] = 0
    return (normIm, taggedIm)

def _tag_image(im, tags, xyShape):
    """
    Converts an array to an RGB image, draws colored boxes.

    'tags' is a list of 2-tuples. The first entry in each tuple is a
    list of localization dicts. The second entry is a 3-tuple RGB
    color vector.     

    """
    import scipy
    
    taggedIm = scipy.zeros((im.shape[0],im.shape[1],3))
    for i in range(3):
        taggedIm[:,:,i] = (im - im.min()) * 1.0 /(im.max() - im.min())
    for (locs, colorTup) in tags:
        for lo in locs:
            (x, y) = (round(lo['x']), round(lo['y']))
            molX = slice(max(x, 0), min(x + xyShape[0], taggedIm.shape[0]))
            molY = slice(max(y, 0), min(y + xyShape[1], taggedIm.shape[1]))
            for i, c in enumerate(colorTup):
                taggedIm[molX.start, molY, i] = c
                taggedIm[molX.stop-1, molY, i] = c
                taggedIm[molX, molY.start, i] = c
                taggedIm[molX, molY.stop-1, i] = c
    return taggedIm

def _localization_filter_string(
    xMax, yMax, zMax, piezoMax=0,
    xMin=0, yMin=0, zMin=0, piezoMin=0,
    funcNum=0, correlationMin=0):

    filtStr = (
        "def locFilter%i(loc):\r\n"%(funcNum) +
        "    if 'z_piezo' in loc:\r\n" +
        "        if loc['z_piezo'] > %f:\r\n"%(piezoMax) +
        "            return False\r\n" +
        "        if loc['z_piezo'] < %f:\r\n"%(piezoMin) +
        "            return False\r\n" +
        "    if loc['x'] > %f:\r\n"%(xMax) + 
        "        return False\r\n" +
        "    if loc['x'] < %f:\r\n"%(xMin) +
        "        return False\r\n" +
        "    if loc['y'] > %f:\r\n"%(yMax) + 
        "        return False\r\n" +
        "    if loc['y'] < %f:\r\n"%(yMin) +
        "        return False\r\n" +
        "    if loc['z'] > %f:\r\n"%(zMax) + 
        "        return False\r\n" +
        "    if loc['z'] < %f:\r\n"%(zMin) +
        "        return False\r\n" +
        "    if loc['qual'] < %f:\r\n"%(correlationMin) +
        "        return False\r\n" +
        "    return True\r\n\r\n")
    return filtStr

def _linking_filter(loc1, loc2):
    if loc1['x'] - loc2['x'] > 3:
        return False
    if loc1['x'] - loc2['x'] < -3:
        return False
    if loc1['y'] - loc2['y'] > 3:
        return False
    if loc1['y'] - loc2['y'] < -3:
        return False
    if loc1['z'] - loc2['z'] > 10:
        return False
    if loc1['z'] - loc2['z'] < -10:
        return False
    return True

def _fiducial_filter(matches):
    if len(matches) > 10:
        return True
    return False

def new_palm(
    images=None,
    imFolder=None,
    im_format='tif',
    custom_im_load=None,
    im_xy_shape=None,
    im_z_positions=None,
    calImages=None,
    calFolder=None,
    cal_format='tif',
    custom_cal_load=None,
    cal_xy_shape=None,
    filename_prefix='',
    save_filename='palm_acquisition.pkl',
    candidates_filename='palm_candidates',
    localizations_filename='palm_localizations',
    linked_localizations_filename='palm_localizations_linked',
    fiducial_filter_filename='palm_selected_fiducials.py',
    particles_filename='palm_particles',
    calibration=None):
    """Creates a new instance of the Palm_3d class. See the docstring
    for the Palm_3d class for details.

    If there's already a saved instance of the Palm_3d class with the
    same 'save_filename', prompts the user to load that instance
    instead. Returns either the old or the new Palm_3d instance.

    If you edit the code for new_palm or Palm_3d, make sure their
    argument lists match!

    """
    inputArguments = locals()
    import os

    """Check if there's a palm object in the current directory:"""
    saveName = filename_prefix + save_filename
    if saveName in os.listdir(os.getcwd()):
        print (
            saveName + " already exists. Remove? y/[n]?"),
        removeOld = raw_input()
        if removeOld != 'y':
            oldPalm = load_palm(saveName)
            print "New Palm_3d object not created."
            print "Old Palm_3d object re-loaded."
            return oldPalm
        else:
            print "Removing old " + saveName
            os.remove(saveName)
    return Palm_3d(**inputArguments)

def load_palm(fileName = 'palm_acquisition.pkl', verbose=True):
    """
    Loads a saved palm acquisition.

    If 'imFolder' or 'calFolder' can't be found (perhaps because the
    acquisition was processed on a different machine), prompts the
    user to pick image and calibration folders.

    """
    import os, sys, cPickle, shelve, types
    if verbose: print "Loading..."
    palm_acquisition = cPickle.load(open(fileName, 'rb'))
    for f in range(2):
        attr = ('imFolder', 'calFolder')[f]
        fol = getattr(palm_acquisition, attr, None)
        if not os.path.exists(fol):
            import Tkinter, tkFileDialog
            tkroot = Tkinter.Tk()
            tkroot.withdraw()
            folName = ("n image", " calibration")[f]
            print ("Can't find folder " + fol + ".\nChoose a" + folName
                   + " folder.")
            setattr(palm_acquisition, attr, os.path.relpath(
                tkFileDialog.askdirectory(
                    title=("Choose a" + folName + " folder"))))
            tkroot.destroy()
    if hasattr(palm_acquisition, 'stored_histograms'):
        print "Reformatting old-style stored histograms..."
        cPickle.dump(
            palm_acquisition.stored_histograms,
            open(os.path.join(
                palm_acquisition.imFolder, 'stored_histograms.pkl'), 'wb'),
            protocol=2)
        delattr(palm_acquisition, 'stored_histograms')
        palm_acquisition.save()
    """OK, finally deprecating the block of legacy code to speed loading:"""
##    """LEGACY. THIS SHOULD NO LONGER BE NECESSARY:
##    Check for old-style analysis, updating from old-style to
##    new-style data if required:"""
##    for attr in ('candidates_filename', 'localizations_filename',
##                 'linked_localizations_filename', 'particles_filename'):
##        if hasattr(palm_acquisition, attr):
##            oldShelfName = getattr(palm_acquisition, attr)
##            if verbose: print attr + ":", oldShelfName
####            print "Exists:", os.path.exists(os.path.join(
####                palm_acquisition.imFolder, oldShelfName))
##            newShelfName = oldShelfName + "_NEW"
##            if os.path.exists(os.path.join(
##                palm_acquisition.imFolder, oldShelfName)):
##                oldShelf = shelve.open(os.path.join(
##                    palm_acquisition.imFolder, oldShelfName), protocol=2)
####                print "'0' in old shelve:", ('0' in oldShelf)
##                if '0' in oldShelf:
##                    if isinstance(oldShelf['0'], types.ListType):
##                        ##New-style data, do nothing
####                        print "Not converting."
##                        oldShelf.close()
##                        oldShelf=None
##                    elif isinstance(oldShelf['0'], types.DictType):
##                        """Old-style data. Convert into new style:"""
##                        print "Converting", oldShelfName, "to", newShelfName
##                        newShelf = shelve.open(os.path.join(
##                            palm_acquisition.imFolder,
##                            newShelfName), protocol=2)
##                        locNum = 0
##                        loc = oldShelf['0']
##                        print oldShelf['num'], "localizations."
##                        for imNum in xrange(len(palm_acquisition.images)):
##                            locList = []
##                            while loc['image_num'] == imNum:
##                                locList.append(dict(loc))
##                                locNum += 1
##                                loc = oldShelf.get( #Default breaks loop
##                                    repr(locNum), {'image_num': None})
##                            newShelf[repr(imNum)] = locList
##                            if ((imNum%50 == 0) or
##                                (imNum == len(palm_acquisition.images)-1)):
##                                sys.stdout.write(
##                                    "\rCopying key: %s"%(locNum - 1))
##                                sys.stdout.flush()
##                                newShelf.sync()
##                                oldShelf.sync()
##                        print ""
##                        for key in oldShelf.keys():
##                            try:
##                                int(key)
##                            except ValueError:
##                                print "Copying key:", key
##                                newShelf[key] = oldShelf[key]
##                                oldShelf.sync()
##                                newShelf.sync()
##                        newShelf.close()
##                        newShelf = None
##                        oldShelf.close()
##                        oldShelf=None
##                        os.rename(os.path.join(palm_acquisition.imFolder,
##                                               oldShelfName),
##                                  os.path.join(palm_acquisition.imFolder,
##                                               "OLD_" + oldShelfName))
##                        os.rename(os.path.join(palm_acquisition.imFolder,
##                                               newShelfName),
##                                  os.path.join(palm_acquisition.imFolder,
##                                               oldShelfName))
##                    else:
##                        print "Neither a dict nor a list!"
##                        print "Keys:", oldShelf.keys()
##                        oldShelf.close()
##                        oldShelf=None
    return palm_acquisition

class Palm_3d:
    """
    3d localization of single fluorescent molecules and nanoparticles
    based on experimental calibration.

    The Palm_3d class has the following attributes:

    ***'images', 'imFolder', 'im_format', 'im_xy_shape': A list of
    image names 'images' in the folder 'imFolder', and the image
    format 'im_format'. 'im_format' can currently be 'tif' for
    micromanager TIF files, 'raw' for Andor DAT files, or 'custom' for
    a user-defined image loading function. Both 'tif' and 'raw' assume
    16-bit Intel byte-order unsigned integer data. If 'raw' is chosen,
    the user must also specify 'im_xy_shape', a 2-tuple giving the
    XY-size in pixels of one image in the DAT file.

    'custom_im_load'
    In addition to a default formats, 'im_format' can be 'custom', in
    which case the user must also specify a function 'custom_im_load' that
    returns a 2d scipy array given a name from 'images'. Each 2d array
    returned by 'im_load' is a 2d image taken in a typical PALM
    acquisition. im_load('im_34.tif')[3,4] is x-pixel=3, y-pixel=4 of
    image 'im_34.tif', for example.


    If 'raw' is chosen and there are multiple images per DAT file,
    then we use a small hack to specify image number. Each entry in
    'images' for a DAT file appends a star '*' and an integer to the
    end of the filename. The star is a separator, the integer is the
    image number. For example, if you had two PALM images in
    spool.dat, followed by one palm image in spool_2.dat, you would
    use:
        images = ['spool.dat*0', spool.dat*1', 'spool_2.dat*0']

    ***'im_z_positions':
    A dict with the entries of 'images' as keys. The value at each key
    is the z-piezo position at which the image was taken. The units of
    z-position are pixels, so their size in nanometers is determined
    by the calibration stack.

    ***'calImages', 'calFolder', 'cal_format', 'custom_cal_load',
    'cal_xy_shape':
    The same as 'images', 'imFolder', 'im_format', 'custom_im_load',
    'im_xy_shape', as defined above, except for the calibration images
    rather than PALM images.

    ***'calibration':
    The calibration images are processed into a 3d scipy array
    'calibration'. 'calibration' is a stack of 2d images, representing
    a single fluorescent molecule imaged at various known z-positions.
    'calibration[3,4,5]' would correspond to x-pixel=3, y-pixel=4 of
    the 5th image in the calibration z-stack. Alternatively, the user
    can specify 'calibration' to use a pre-existing calibration stack.

    ***'filename_prefix':
    A string that is prepended to save, candidates, and localization
    files produced by the Palm_3d object. Useful to avoid overwriting
    if running multiple localization scripts in the same directory.

    ***'save_filename':
    A string. What filename to use when saving the palm acquisition.

    ***'candidates_filename', 'localizations_filename',
    'linked_localizations_filename', 'particles_filename':
    The PALM images are processed into a 'shelve' of lists of dicts.
    Each dict in a list represents one localization of a particle.
    Each list in the shelve contains all the localizations from a
    single image.  As various properties of each localization are
    calculated, the corresponding entries of the dict are filled in.
    The 'shelve' object is stored on disk using the corresponding filename.

    Mandatory entries in a 'localizations' dict:
    *'image_num':
    A unique integer indicating the ordering of the images, important
    for linking adjacent frames.
    * 'image_name':
    A string indicating which image the localization first appears in.
    The image corresponding to localization 'loc' is returned by
    im_load(loc['image_name'])
    The following should be always be true:
    images[loc['image_num']] == loc['image_name']
    
    * 'x_slice':
    * 'y_slice':
    The (x,y) slice of the region of im_load(loc['image_name']) in
    which the particle appears. The xy-size of this region is the same
    as the xy size of 'calibration'.

    Optional entries in a 'localizations' dict:

    * 'correlations':
    A list of 3d scipy arrays, which are xyz-resampled subregions of the
    correlation of the particle image with 'calibration'. Each entry is
    sampled more finely than the previous entry. Each xyz-region is
    roughly centered on the maximum of the previous entry. 'correlations'
    is likely to be large, and is typically not stored unless debugging.

    * 'x':
    * 'y':
    * 'z':
    * 'qual':
    Floating-point estimated locations of the glowing particle, in units
    of pixels. 'qual' is a measure of the quality of this estimate.
    loc['qual'] = correlations[-1].max()

    *'later_image_names':
    A list of strings, indicating later images in which the same particle
    is also localized. Useful for averaging several frames in which the
    same particle appears. Be careful that the particle does not have time
    to drift over the course of these frames!

    *'single_image_localizations':
    A list of dicts containing the results of processing each image of the
    particle separately, instead of averaging frames.

    *'bg_image_names':
    A list of strings, indicating images which immediately precede or
    follow the appearance of a particle. Useful for subtracting
    background, if the background is steady enough.

    *'avg_counts':
    Floating-point average value of the particle images. Useful for
    filtering out dim or saturated particles.

    *'avg_counts_above_bg':
    Floating-point average value of the particle images, minus the
    average value of the background frames. Useful for filtering out dim
    or saturated particles.

    """
    def __init__(
        self,
        images=None,
        imFolder=None,
        im_format='tif',
        custom_im_load=None,
        im_xy_shape=None,
        im_z_positions=None,
        calImages=None,
        calFolder=None,
        cal_format='tif',
        custom_cal_load=None,
        cal_xy_shape=None,
        filename_prefix='',
        save_filename='palm_acquisition.pkl',
        candidates_filename='palm_candidates',
        localizations_filename='palm_localizations',
        linked_localizations_filename='palm_localizations_linked',
        fiducial_filter_filename='palm_selected_fiducials.py',
        particles_filename='palm_particles',
        calibration=None
        ):
        import os, types

        """Define how to load PALM images:"""
        if isinstance(images, types.ListType):
            self.images = images
        else:
            raise UserWarning("'images' should be a list of images.")
        if imFolder is None:
            imFolder = os.getcwd()
        if not os.path.exists(str(imFolder)):
            import Tkinter, tkFileDialog
            tkroot = Tkinter.Tk()
            tkroot.withdraw()
            print ("Can't find image folder " + imFolder +
                   ".\nChoose an image folder.")
            imFolder = os.path.relpath(tkFileDialog.askdirectory(
                title=("Choose an image folder")))
            tkroot.destroy()
        self.imFolder = imFolder
        self.im_format = im_format
        self.im_xy_shape = im_xy_shape
        if self.im_format == 'raw' and self.im_xy_shape is None:
            raise UserWarning("im_xy_shape must be set if im_format='raw'")
        if self.im_format == 'sif' or cal_format == 'sif':
            self.sifInfo = {} #To cache SIF header info
        if im_format == 'custom':
            """Let the user define a custom palm image loading
            function. This seems to break saving, but just in case:"""
            if isinstance(custom_im_load, types.FunctionType):
                self.custom_im_load = custom_im_load
            else:
                raise UserWarning("Custom image loader not a function.")
        if isinstance(im_z_positions, types.DictType) or im_z_positions is None:
            self.im_z_positions = im_z_positions
        else:
            raise UserWarning("'im_z_positions' should be a dict or 'None'." +
                              "See the palm_3d documentation for examples.")
        """Define how to load calibration images:"""
        self.calImages = calImages
        if calFolder is None:
            calFolder = os.getcwd()
        if not os.path.exists(str(calFolder)):
            import Tkinter, tkFileDialog
            tkroot = Tkinter.Tk()
            tkroot.withdraw()
            print ("Can't find calibration folder " + calFolder +
                   ".\nChose a calibration folder.")
            calFolder = os.path.relpath(tkFileDialog.askdirectory(
                title=("Choose a calibration folder")))
            tkroot.destroy()
        self.calFolder = calFolder
        self.cal_format = cal_format
        self.cal_xy_shape = cal_xy_shape
        if self.cal_format == 'raw' and self.cal_xy_shape is None:
            raise UserWarning("cal_xy_shape must be set if cal_format='raw'")            
        if cal_format =='custom':
            """Let the user define a custom calibration image loading
            function. This seems to break saving, but just in case:"""
            if isinstance(custom_cal_load, types.FunctionType):
                self.custom_cal_load = custom_cal_load
            else:
                raise UserWarning("Custom calibration loader not a function")
        """The filenames of our persistent datatypes:"""
        self.save_filename = filename_prefix + save_filename
        self.candidates_filename = filename_prefix + candidates_filename
        self.localizations_filename = filename_prefix + localizations_filename
        self.linked_localizations_filename = (
            filename_prefix + linked_localizations_filename)
        self.fiducial_filter_filename = ( ##Tricky to make this a path
            filename_prefix + fiducial_filter_filename)
        self.particles_filename = filename_prefix + particles_filename
        """Finally, if we already have calibration or localization:"""
        self.calibration = calibration
        return None

    im_load = _im_load
    cal_load = _cal_load
    get_xy_slices = _get_xy_slices    

    def save(self, fileName=None, overwriteWarning=True):
        """
        Saves a palm acquistion, to be loaded later with
        palm_3d.load_palm().

        If you accidentally corrupt or destroy a saved acquisition and
        want it back, there's a good chance you can reproduce the
        acquisition. quickly by re-running whatever script produced it
        in the first place. Many expensive operations (such as
        candidate selection or localization) leave behind files, so
        you might not have to repeat these operations.
        """
        import os, cPickle
        
        if fileName is None:
            fileName = self.save_filename
        if os.path.isfile(fileName) and overwriteWarning:
            print "Saving ", fileName, "..."
            print "The file '" + fileName + "'",
            print "already exists. Overwrite? y/[n]:",
            overwrite_acquisition = raw_input()
            if overwrite_acquisition == 'y':
                os.remove(fileName)
            else:
                print "Save cancelled.\n"
                return None
        print "Saving palm acquisition as " + fileName + " ..."
        self.version = version
        try:
            cPickle.dump(self, open(fileName, 'wb'), protocol=2)
        except IOError:
            ##http://support.microsoft.com/default.aspx?scid=kb;en-us;899149
            cPickle.dump(self, open(fileName, 'w+b'), protocol=2)
        return None

    def load_calibration(
        self,
        calibrationRepetitions=1,
        calibrationImagesPerPosition=1,
        xSlice = None,
        ySlice = None,
        zSlice = 'all',
        smoothing_sigma=(1, 1, 5),
        promptForSave=True,
        promptForInspection=True
        ):
        import os, cPickle
        import scipy, pylab
        """
        Loads a calibration stack for 3d localization.

        One good way to construct this information is to put a gold
        nanoparticle on your microscope slide and move the slide in
        small z-steps with a piezo while taking pictures. Be careful
        to let the piezo ring down after each motion, before taking a
        picture! If the image of the nanoparticle changes noticably
        with each movement, we can use this information to localize
        other glowing particles in 3d.

        calibrationRepetitions is the number of consecutive
        calibration stacks taken, each assumed identical. This is so
        you can take several calibration stacks in a row and average
        them together. Watch out for thermal drift if you do this!

        calibrationImagesPerPosition is the number of consecutive images
        taken at each calibration position, in a single calibration stack.
        This is so you can average several adjacent shots and not have to
        worry about thermal drift as much, maybe.

        """
        import sys

        if self.calibration is not None:
            print "Calibration data already loaded. Reload? y/[n]:",
            reload_calibration = raw_input()
            if reload_calibration != 'y':
                print "Using existing calibration data instead.\n"
                return None

        print "Calibration folder: " + self.calFolder
        if 'calibration.pkl' in os.listdir(self.calFolder):
            print 'Looks like we already processed calibration data.'
            print 'Loading .pkl instead...'
            self.calibration = cPickle.load(
                open(os.path.join(self.calFolder, 'calibration.pkl'),'rb'))
            if 'calibration_data.pkl' in os.listdir(self.calFolder):
                self.calibrationData = cPickle.load(open(
                    os.path.join(self.calFolder, 'calibration_data.pkl'),'rb'))
        else:            
            if xSlice == None or ySlice == None:
                xSlice, ySlice = self.get_xy_slices()
            if zSlice == 'all':
                zSlice = slice(0, len(self.calImages))
            print 'xSlice:', xSlice
            print 'ySlice:', ySlice
            print 'zSlice:', zSlice
            print 'Loading...'
            calibration = scipy.zeros((
                xSlice.stop - xSlice.start,
                ySlice.stop - ySlice.start,
                len(self.calImages)))
            for c, calIm in enumerate(self.calImages):
                calibration[:, :, c] = self.cal_load(calIm)[xSlice,ySlice]
                sys.stdout.write("\rFrames loaded:%7d/%d"%(
                    c+1, len(self.calImages)))
                sys.stdout.flush()
            calibration = calibration[:,:,zSlice]
            print "\n\nCalibration max-min:",
            print calibration.max() - calibration.min()
            if calibrationRepetitions is not 1:
                ##Average stacks:
                print "Calibration shape:", calibration.shape
                print "Averaging across repetitions..."
                calibration = calibration.reshape((
                    calibration.shape[0],
                    calibration.shape[1],
                    calibrationRepetitions,
                    calibration.shape[2]/calibrationRepetitions,
                    ))
                print "Calibration max. std. dev: ", calibration.std(2).max()
                calibration = calibration.mean(2)
                print "Calibration shape:", calibration.shape, "\n"
            if calibrationImagesPerPosition is not 1:
                ##Average shots:
                print "Calibration shape:", calibration.shape
                print "Averaging across consecutive shots..."
                calibration = calibration.reshape((
                    calibration.shape[0],
                    calibration.shape[1],
                    calibration.shape[2]/calibrationImagesPerPosition,
                    calibrationImagesPerPosition
                    ))
                print "Calibration max. std. dev: ", calibration.std(3).max()
                calibration = calibration.mean(3)
                print "Calibration shape:", calibration.shape, "\n"
            if smoothing_sigma is not None:
                from scipy.ndimage import gaussian_filter
                print "Smoothing with sigma=...", smoothing_sigma
                calibration = gaussian_filter(
                    calibration, sigma=smoothing_sigma)
            calibrationData = {}
            calibrationData[
                'calibrationRepetitions'] = calibrationRepetitions
            calibrationData[
                'calibrationImagesPerPosition'] = calibrationImagesPerPosition
            calibrationData['xSlice'] = xSlice
            calibrationData['ySlice'] = ySlice
            calibrationData['zSlice'] = zSlice
            calibrationData['smoothing_sigma'] = smoothing_sigma
            try:
                cPickle.dump(
                    calibration,
                    open(os.path.join(self.calFolder, 'calibration.pkl'),'wb'),
                    protocol=2)
                cPickle.dump(
                    calibrationData,
                    open(os.path.join(
                        self.calFolder, 'calibration_data.pkl'),'wb'),
                    protocol=2)
            except IOError:
                ##http://support.microsoft.com/default.aspx?scid=kb;en-us;899149
                cPickle.dump(
                    calibration,
                    open(os.path.join(self.calFolder, 'calibration.pkl'),'w+b'),
                    protocol=2)
                cPickle.dump(
                    calibrationData,
                    open(os.path.join(
                        self.calFolder, 'calibration_data.pkl'),'w+b'),
                    protocol=2)
            self.calibration = calibration
            self.calibrationData = calibrationData

        if promptForInspection:
            print "Inspect calibration? [y]/n:",
            inspectCal = raw_input()
            if inspectCal != 'n':
                self.inspect_calibration(showResults=True, saveResults=True)
        if promptForSave:
            self.save()
        return None

    def inspect_calibration(self, showResults=False, saveResults=False):
        """
        Correlates a calibration stack against itself, to quantify how
        similar or different each calibration slice is.

        A nice thin diagonal band in the output 'calXC.png' probably
        implies a useful calibration stack, because the image of a
        nanoparticle will change measurably with each small
        displacement in z. A region where the diagonal band gets
        fatter implies a region where the image of a nanoparticle
        changes slowly as the particle moves in z, implying poor
        z-localization for particles that look like this.

        """
        """Construct artificial 'images' and 'im_load':"""
        print "Checking calibration stack cross-correlations..."
        calibrationCrossterms = self.localize_individual_images(
            self.calibration, initial_upsample=1.)
        
        if showResults or saveResults:
            import pylab
            pylab.figure()
            pylab.imshow(calibrationCrossterms, interpolation='nearest')
            pylab.colorbar()
            pylab.title(
                'Calibration cross-correlations. Watch out for big red patches!')
            if showResults: pylab.gcf().show()
            if saveResults: pylab.savefig('calXC.png')
            print "Plotting calibration slices..."
            figList = plot_slices(self.calibration, showFigs=showResults)
            for i, fig in enumerate(figList):
                if saveResults: fig.savefig('cal%i.png'%(i+1))
            print "Calibration slices. Hit enter."
            raw_input()
            pylab.close('all')
        return calibrationCrossterms

    def images_to_candidates(
        self,
        numSTD=4,
        numSTD_changed=None,
        showSteps=False
        ):
        """
        Populates 'localizations' with suspected particles from image
        data.

        You should have already defined a calibration stack with
        load_calibration() (or setting self.calibration by hand)
        before calling this method.

        numSTD is how bright a 'bright' candidate molecule must be,
        compared to an estimate of the average fluctuations of the
        image, with low- and high-spatial-frequency background
        filtered out.

        numSTD_changed is how much a 'birth' or 'death' candidate
        molecule must change, compared to estimated fluctuations.

        """
        import sys, shelve
        import scipy

        if os.path.isfile(os.path.join(
            self.imFolder, self.candidates_filename)):
            print "The file '"+self.candidates_filename+"' already exists."
            print "Looks like we already selected candidate particles.",
            print "Overwrite? y/[n]: ",
            overWrite = raw_input()
            if overWrite != 'y':
                print "Using old candidate particles file. Be careful!\n"
                return None
            else:
                print "Removing old candidate particles."
                os.remove(os.path.join(self.imFolder, self.candidates_filename))
        candidates = shelve.open(os.path.join(
            self.imFolder, self.candidates_filename), protocol=2)
        """Keep track of how many candidate particles:"""
        candidates['num'] = 0
        candidates['num_brights'] = 0
        candidates['num_births'] = 0
        candidates['num_deaths'] = 0

        print "Test selection parameters? [y]/n:",
        testParams = raw_input()
        if testParams != 'n':
            import pylab
            (numSTD, numSTD_changed) = self.inspect_candidate_selection(
                numSTD=numSTD, numSTD_changed=numSTD_changed)
            pylab.close('all')
            print "Done testing. Parameters chosen:"
            print "numSTD:", numSTD
            print "numSTD_changed:", numSTD_changed
        print "Locating candidate particles..."
        candidates['numSTD'] = numSTD
        candidates['numSTD_changed'] = numSTD_changed
        if showSteps:
            import pylab
            pylab.close('all')
            pylab.figure()
        nums = {'num': 0, 'num_births': 0, 'num_deaths': 0, 'num_brights': 0}
        for image_num, image_name in enumerate(self.images):
            if image_num is 0:
                previousFrame = None
            else:
                previousFrame = 1.0 * imageIn
            ##Load a frame and smooth it:
            imageIn = _detection_filter(self.im_load(image_name))
            (brights, births, deaths) = _image_to_molecule_locations(
                imageIn=imageIn,
                image_num=image_num,
                xyShape=self.calibration.shape[0:2],
                previousFrame=previousFrame,
                numSTD=numSTD,
                numSTD_changed=numSTD_changed,
                showResults=showSteps)
            if showSteps:
                pylab.gcf().show()
                print "Continue? [y]/n:",
                keepGoing = raw_input()
                pylab.clf()
                if keepGoing == 'n':
                    showSteps = False
                    pylab.close('all')
            """The order births, deaths, brights is important for linking:"""
            candList = []
            for cands in (births, deaths, brights):
                if cands is births:
                    flagStr = 'birth_flag'
                    numStr = 'num_births'
                if cands is deaths:
                    flagStr = 'death_flag'
                    numStr = 'num_deaths'
                if cands is brights:
                    flagStr = 'bright_flag'
                    numStr = 'num_brights'
                for (molX, molY) in cands:
                    candList.append({
                        'image_name': image_name,
                        'x_slice': molX,
                        'y_slice': molY,
                        'image_num': image_num,
                        flagStr: True})
                    nums['num'] += 1
                    nums[numStr] += 1
            candidates[repr(image_num)] = candList
            if (image_num%50 == 0) and (showSteps is False):
                sys.stdout.write(
                    "\rImages processed:%7d/%d. Candidates found: %7d"%(
                        image_num+1, len(self.images), nums['num']))
                sys.stdout.flush()
                candidates.sync()
        for k, v in nums.items():
            candidates[k] = v
        sys.stdout.write(
            "\rImages processed:%7d/%d. Candidates found: %7d"%(
                image_num+1, len(self.images), candidates['num']))
        print '\nDone selecting particles.\n'
        print "%i bright candidates found"%(candidates['num_brights'])
        print "%i birth candidates found"%(candidates['num_births'])
        print "%i death candidates found"%(candidates['num_deaths'])
        candidates.close()
        candidates=None #To prevent a possible weirdo error message.
        return None

    def inspect_candidate_selection(
        self, imageNum=0, numSTD=4, numSTD_changed=None):
        """
        Show the steps used to select candidates.

        Useful for tuning the numSTD and numSTD_changed parameters, to
        ensure we catch most molecules and not too much garbage.

        """
        import pylab

        pylab.close('all')
        pylab.figure()
        while True:
            if imageNum >= len(self.images):
                print "There are only %i images to show."%(len(self.images))
                imageNum = len(self.images) - 1
            pylab.suptitle(" \n\n\nDrawing...")
            pylab.ioff()
            imageIn = self.im_load(self.images[imageNum])
            filteredImage = _detection_filter(imageIn)
            if imageNum > 0:
                previousFrame = self.im_load(self.images[imageNum - 1])
                filteredPreviousFrame = _detection_filter(previousFrame)
            else:
                previousFrame = None
                filteredPreviousFrame = None
            pylab.clf()
            _image_to_molecule_locations(
                imageIn=filteredImage,
                image_num=imageNum,
                xyShape=self.calibration.shape[0:2],
                previousFrame=filteredPreviousFrame,
                unfilteredImage=imageIn,
                unfilteredPreviousFrame=previousFrame,
                numSTD=numSTD,
                numSTD_changed=numSTD_changed,
                showResults=True)
            pylab.gcf().show()
            pylab.ion()
            print "Continue, jump images, new parameters,",
            print "or finished testing? [c]/j/n/f:",
            cmd = raw_input()
            if cmd == 'f':
                break
            elif cmd == 'n':
                while True:
                    print "numSTD:",
                    numSTD = raw_input()
                    print "numSTD_changed:",
                    numSTD_changed = raw_input()
                    try:
                        numSTD = float(numSTD)
                        numSTD_changed = float(numSTD_changed)
                        break
                    except ValueError:
                        print "Type a number, hit return."
            elif cmd == 'j':
                while True:
                    print "New image number:",
                    imageNum = raw_input()
                    try:
                        imageNum = int(imageNum)
                        break
                    except ValueError:
                        print "Type an integer, hit return."
            else:
                imageNum += 1
        return (numSTD, numSTD_changed)

    def localize_candidates(
        self,
        initial_upsample=0.35,
        saveCorrelations=False,
        showInterpolation=False,
        linkedInput=False,
        processBirths=True,
        processDeaths=True,
        retainEdgeFlags=False,
        searchPastEdges=True,
        promptForInspection=True,
        verbose=True
        ):
        """
        Matches a particle image to the closest image in a calibration
        stack, estimating (possibly subpixel) xyz shifts which
        maximize alignment.

        This function can process candidates into localizations, (with
        'linkedInput=False') or linked localizations into particles
        (with 'linkedInput=True').

        """
        import os, sys, shelve
        import scipy
        from scipy.fftpack import fftn

        if linkedInput:
            candidates_filename = self.linked_localizations_filename
            localizations_filename = self.particles_filename
            print "Relocalizing linked localizations."
        else:
            candidates_filename = self.candidates_filename
            localizations_filename = self.localizations_filename
            if verbose: print "Localizing candidates."

        if not os.path.isfile(os.path.join(self.imFolder, candidates_filename)):
            print "Particle candidates file '" + candidates_filename + "'",
            print " not found."
            print "Locate the file, or run ",
            if linkedInput:
                print "link_localizations() first."
            else:
                print "images_to_candidates()' first."
            return None
        candidates = shelve.open(os.path.join(
            self.imFolder, candidates_filename), protocol=2)        

        if os.path.isfile(os.path.join(self.imFolder, localizations_filename)):
            print "The file '"+localizations_filename+"' already exists."
            print "Looks like we already localized",
            if linkedInput:
                print "linked localizations."
            else:
                print "candidate particles.",
            print "Overwrite? y/[n]: ",
            overWrite = raw_input()
            if overWrite != 'y':
                print "Using old particle localizations file. Be careful!\n"
                return None
            else:
                print "Removing old localizations."
                os.remove(os.path.join(self.imFolder, localizations_filename))
        localizations = shelve.open(os.path.join(
            self.imFolder, localizations_filename), protocol=2)
        """Keep track of how many localized particles."""
        localizations['num'] = 0
        localizations['initial_upsample'] = initial_upsample

        smoothingWindow = _smoothing_window(self.calibration.shape)
        calibrationFT = ( #FFT normalization is funny
            scipy.sqrt(self.calibration.shape[0] * self.calibration.shape[1]) *
            fftn(normalize_slices(
                fftn(self.calibration, axes=(0,1)) *
                scipy.atleast_3d(smoothingWindow)), axes=(2,)))
        """Compute cross-correlations between each image and each slice of
        the calibration stack:"""
        if verbose: print "Computing cross-correlations..."
        whichCandidate = -1
        for imageNum, imageName in enumerate(self.images):
            """locList is a COPY. We don't want to edit 'candidates'"""
            locList = candidates[repr(imageNum)]
            if imageNum == 0:
                lastImage = None
            else:
                lastImage = 1.0 * fullImage ##A copy
            fullImage = self.im_load(imageName)
            if imageNum%20 == 0:
                candidates.sync()
                localizations.sync()                
            for whichLoc, loc in enumerate(locList):
                whichCandidate += 1
                if whichCandidate%20 == 0:
                    if verbose:
                        sys.stdout.write("\rCandidates processed:%7d/%d"%(
                            whichCandidate+1, candidates['num']))
                        sys.stdout.flush()
                if ((('birth_flag' in loc) and not processBirths) or
                    (('death_flag' in loc) and not processDeaths)):
                    continue
                (loc, locImage) = self._localization_image(
                    localization=loc, fullImage=fullImage, lastImage=lastImage)
                """Normalize and FFT:"""
                locImageFT = (scipy.sqrt(
                    self.calibration.shape[0] * ##FFT normalization is funny
                    self.calibration.shape[1]) *
                    normalize_slices(fftn(locImage) * smoothingWindow))
                """One image, correlated with all the calibrations:"""
                locStackFT = scipy.transpose(scipy.tile(
                    locImageFT, (calibrationFT.shape[2], 1, 1)), (1, 2, 0))
                correlationsFT = scipy.conjugate(locStackFT) * calibrationFT
                correlationsHist, coordsHist = [], []
                """Initial coarse, cheap search for the maximum:"""
                num = scipy.ceil(initial_upsample *
                                 scipy.array(correlationsFT.shape)).astype(int)
                (correlations, coords) = dft_resample_n(
                    x_ft=correlationsFT, num=num,
                    t=[None]*3, axes=range(3), window=[None]*3, ifft=True)
                correlationsHist.append(1.0 * correlations)
                coordsHist.append(list(coords))
                """Now, guess the particles's position to within 4 pixels:"""
                maxIndex = correlations.argmax()
                (xSh, ySh, zSh) = scipy.unravel_index(
                    maxIndex, correlations.shape)
                xyzShift = scipy.array((
                    coords[0][xSh], coords[1][ySh], coords[2][zSh]))
                """Use Fourier interpolation to guess the subpixel x, y, and z
                shifts which would maximize correlation:"""
                zoomLevels = [2, 20] ##Choose carefully!
                samples = scipy.array((11,11,41)) ##Same here.
                window = [None, None, 'hann']
                whichZoom = 0
                zIndexShift = 0
                numIndexShifts = 0 #Non-unique maxima can be pesky
                while True:
                    zoom = zoomLevels[whichZoom]
                    num = (zoom * scipy.array(correlationsFT.shape)).astype(int)
                    start_index = ((zoom * xyzShift - samples // 2) //
                                   1).astype(int)
                    start_index[2] += zIndexShift
                    (correlations, coords) = dft_resample_n(
                        x_ft=correlationsFT, num=num,
                        start_index=start_index, samples=samples,
                        t=[None]*3, axes=range(3), window=window)
                    maxIndex = correlations.argmax()
                    (xSh, ySh, zSh) = scipy.unravel_index(
                        maxIndex, correlations.shape)
                    xyzShift = scipy.array((
                        coords[0][xSh], coords[1][ySh], coords[2][zSh]))
                    if ((int(zSh) == 0) or
                        (int(zSh) == correlations.shape[2] - 1)):
                        if searchPastEdges and (numIndexShifts < 5):
                            """Redo the search at the same zoom level, but
                            move your start_index by almost a window:"""
                            if int(zSh) == 0:
                                zIndexShift -= (samples[2] - 2)
                            else:
                                zIndexShift += (samples[2] - 2)
                            numIndexShifts += 1
                            continue
                        else:
                            """Give up, move on to the next zoom level."""
                            loc['edge_flag'] = True
                    correlationsHist.append(1.0*correlations)
                    coordsHist.append(list(coords))
                    whichZoom += 1
                    zIndexShift = 0
                    if whichZoom >= len(zoomLevels):
                        break
                if 'edge_flag' in loc:
                    if loc['edge_flag'] and not retainEdgeFlags:
                        locList[whichLoc]=loc
                        continue
                """Correlation shifts are periodic:"""
                wrapLength = scipy.array(correlationsFT.shape[0:2] + (0,))//2
                xyzShift = ((xyzShift + wrapLength)%correlationsFT.shape -
                            wrapLength)
                loc['x'] = loc['x_slice'].start - xyzShift[0]
                loc['y'] = loc['y_slice'].start - xyzShift[1]
                loc['z'] = xyzShift[2]
                loc['qual'] = correlations.flat[maxIndex]
                """This is big, don't save it unless you need it:"""
                if saveCorrelations:
                    loc['correlations'] = correlationsHist
                if self.im_z_positions is not None:
                    loc['z_piezo'] = self.im_z_positions[imageName]
                locList[whichLoc] = loc
                if showInterpolation:
                    import pylab
                    from scipy.fftpack import ifftn
                    if whichCandidate > 0:
                        print "Hit enter to continue..."
                        raw_input()
                    pylab.clf()
                    pylab.suptitle('%0.2f %0.2f %0.2f, initial upsample:%0.2f'%(
                        loc['x'], loc['y'], loc['z'], initial_upsample))
                    pylab.subplot(2,1,1)
                    pylab.hold('on')
                    for i, corr in enumerate(correlationsHist):
                        cVSz = []
                        for z in range(corr.shape[2]):
                            cVSz.append(corr[:,:,z].max())
                        pylab.plot(coordsHist[i][2], cVSz, '.')
                    pylab.grid()
                    pylab.subplot(2,4,5)
                    zC = max(int(loc['z']//1), 0)
                    pylab.imshow(
                        self.calibration[:,:,zC], interpolation='nearest')
                    pylab.xlabel("Cal. z=%i"%(zC))
                    pylab.subplot(2,4,6)
                    pylab.imshow(locImage, interpolation='nearest')
                    brightSpot = scipy.array(scipy.unravel_index(
                        locImage.argmax(), locImage.shape))
                    pylab.arrow(
                        x=brightSpot[1],y=brightSpot[0],dx=xyzShift[1],
                        dy=xyzShift[0], ec='w', fc='w')
                    pylab.xlabel("Loc. image")
                    pylab.subplot(2,4,7)
                    pylab.imshow(ifftn(calibrationFT)[:,:,zC].real,
                                 interpolation='nearest')
                    pylab.xlabel("Filtered cal.")
                    pylab.subplot(2,4,8)
                    pylab.imshow(
                        ifftn(locImageFT).real, interpolation='nearest')
                    pylab.arrow(
                        x=brightSpot[1],y=brightSpot[0], dx=xyzShift[1],
                        dy=xyzShift[0], ec='w', fc='w')
                    pylab.xlabel("Filtered shifted im.")
                    pylab.gcf().show()
            """Outside the locList loop now, inside the self.images loop:"""
            if not retainEdgeFlags:
                locList = [loc for loc in locList if 'edge_flag' not in loc]
            localizations[repr(imageNum)] = locList
            localizations['num'] += len(locList)
        if verbose:
            sys.stdout.write("\rCandidates processed:%7d/%d"%(
                whichCandidate+1, candidates['num']))
            print "\nCalibration stack shape: ", calibrationFT.shape
            print "Particle image shape: ", locImageFT.shape
        candidates.close()
        localizations.close()
        if promptForInspection:
            print "Inspect localizations? [y]/n:",
            inspectLocalizations = raw_input()
            if inspectLocalizations != 'n':
                self.inspect_localizations(linkedInput=linkedInput)
        return None

    def _localization_image(
        self, localization, fullImage=None, lastImage=None):
        """
        Load the image. Average multiple frames, if present. Subtract
        background, if present.
        """
        import scipy
        
        loc = dict(localization)
        if fullImage is None:
            fullImage = self.im_load(loc['image_name'])
        """The 1.0* is very important! Slices don't make copies:"""
        locImage = 1.0 * fullImage[loc['x_slice'], loc['y_slice']]
        if ('birth_flag' in loc) or ('death_flag' in loc):
            if lastImage is None:
                lastImage = self.im_load(self.images[loc['image_num'] - 1])
            locImage -= lastImage[loc['x_slice'], loc['y_slice']]
            if 'death_flag' in loc:
                locImage = -1 * locImage
        elif 'later_image_names' in loc:
            for im in loc['later_image_names']:
                locImage += self.im_load(im)[loc['x_slice'], loc['y_slice']]
            locImage = locImage * (1.0/(1. + len(loc['later_image_names'])))
        loc['avg_counts'] = locImage.mean()
        if 'bg_image_names' in loc:
            bgImage = scipy.zeros(locImage.shape)
            for im in loc['bg_image_names']:
                bgImage += self.im_load(im)[loc['x_slice'], loc['y_slice']]
            bgImage = bgImage * 1.0 / (len(loc['bg_image_names']))
            locImage -= bgImage
            loc['avg_counts_above_bg'] = locImage.mean()
        return (loc, locImage)

    def inspect_localizations(
        self, imageNum=0, liveInput=True, linkedInput=False):
        """
        Inspect localization, so you can judge if the algorithm is
        doing a good job.

        Run this after 'localize_candidates()' to make sure nothing
        insane is going on. A bad calibration stack, high background,
        or (god forbid) a bug in the localization algorithm will
        probably cause insane results that would show up in this
        inspection.

        """
        import os, shelve, pprint
        import scipy, pylab

        if not os.path.isfile(os.path.join(
            self.imFolder, self.localizations_filename)):
            print "Localizations file '" + self.localizations_filename + "'",
            print " not found."
            print "Locate the file, or run localize_candidates() first."
            return None
        localizations = shelve.open(os.path.join(
            self.imFolder, self.localizations_filename), protocol=2)
        if 'initial_upsample' in localizations:
            initial_upsample = localizations['initial_upsample']
        else:
            print "No initial upsample stored, using 0.25"
            initial_upsample = 0.25
        if linkedInput:
            if not os.path.isfile(os.path.join(
                self.imFolder, self.particles_filename)):
                print "Particles file '" + self.particles_filename + "'",
                print " not found."
                print "Locate the file, or run",
                print "localize_candidates(linkedInput=True) first."
                return None
            particles = shelve.open(os.path.join(
                self.imFolder, self.particles_filename), protocol=2)
            if 'initial_upsample' in particles:
                if abs(initial_upsample -
                       particles['initial_upsample']) > 0.01:
                    print "Careful: Particle and candidate localization"
                    print " used different initial upsamples. Using %0.2f"%(
                        initial_upsample)
        maxImageNum = len(self.images) - 1
        pylab.close('all')
        pylab.figure()
        while True:
            if imageNum > maxImageNum:
                print "Maximum image number: ", maxImageNum
                imageNum = maxImageNum
            pylab.suptitle("\n\nSearching...")
            locList = localizations[repr(imageNum)]
            brightLocs = [loc for loc in locList if 'bright_flag' in loc]
            birthLocs = [loc for loc in locList if 'birth_flag' in loc]
            deathLocs = [loc for loc in locList if 'death_flag' in loc]
            particleLocs = [] ##Only used if we've linked and relocalized
            if linkedInput:
                particleLocs = particles[repr(imageNum)]
            pylab.suptitle("\n\n"+" "*40+"Drawing...")
            pylab.ioff()
            imageName = self.images[imageNum]
            im = self.im_load(imageName)
            if imageNum > 0:
                imDiff = im - self.im_load(self.images[imageNum - 1])
            else:
                imDiff = scipy.zeros(im.shape)
            im = _tag_image(
                im, [(brightLocs, (1,0,0)), (particleLocs, (1,1,1))],
                self.calibration.shape)
            imDiff = _tag_image(
                imDiff, [(birthLocs, (0,1,0)), (deathLocs, (0,0,1))],
                self.calibration.shape)
            pylab.clf()
            pylab.suptitle("Image %i: "%(imageNum) + imageName)
            for t, imageLocs in enumerate(
                (brightLocs, particleLocs, birthLocs, deathLocs)):
                if t == 1 or t == 3:
                    pass
                else:
                    pylab.subplot(1,2,t//2+1)
                    pylab.imshow((im, imDiff)[t//2], interpolation='nearest')
                for i, loc in enumerate(imageLocs):
                    pylab.text(
                        loc['y'], loc['x'], '%i'%(i), weight='heavy',
                        color=('red', 'white', 'green', 'blue')[t],
                        alpha=scipy.sqrt(loc['qual']))
                    print (("Bright", "Particle", "Birth", "Death")[t] +
                           " localization %i: x=%5.2f y=%5.2f"%(
                               i, loc['x'], loc['y'])),
                    print "z=%5.2f qual=%5.2f image %i"%(
                        loc['z'], loc['qual'], loc['image_num'])
            pylab.gcf().show()
            pylab.ion()
            if not liveInput: break
            print ("\n(c)ontinue, (j)ump images, (p)rint details," +
                   " (v)iew localization,\nor (f)inished inspecting?" +
                   "[c]/j/p/v/f:"),
            cmd = raw_input()
            if cmd == 'f':
                break
            elif cmd == 'j':
                while True:
                    print "New image number:",
                    imageNum = raw_input()
                    try:
                        imageNum = int(imageNum)
                        break
                    except ValueError:
                        print "Type an integer, hit return."
            elif cmd == 'p':
                for i, loc in enumerate(
                    brightLocs + birthLocs + deathLocs + particleLocs):
                    print "\nLocalization %i:"%(i)
                    pprint.pprint(loc)
            elif cmd == 'v':
                print "(b)right, b(i)rth, (d)eath, or (p)article? [b]/i/d/p:",
                bidp = raw_input()
                if bidp == 'i':
                    if len(birthLocs) > 0:
                        imageLocs = birthLocs
                    else:
                        print "No birth localizations this frame."
                        continue
                elif bidp == 'd':
                    if len(deathLocs) > 0:
                        imageLocs = deathLocs
                    else:
                        print "No death localizations this frame."
                        continue
                elif bidp == 'p':
                    if len(particleLocs) > 0:
                        imageLocs = particleLocs
                    else:
                        print "No particle localizations this frame."
                        continue
                else:
                    if len(brightLocs) > 0:
                        imageLocs = brightLocs
                    else:
                        print "No bright localizations this frame."
                        continue
                while True:
                    print "Localization number to view:",
                    i = raw_input()
                    try:
                        i = int(i)
                    except ValueError:
                        print "Type an integer, hit return."
                        continue
                    try:
                        loc = imageLocs[i]
                        break
                    except IndexError:
                        print "Valid numbers are", range(len(imageLocs))
                pprint.pprint(loc)
                (trash, testIm) = self._localization_image(loc)
                ius = 1.0 * initial_upsample #Temporary copy
                while True:
                    pylab.ioff()
                    self.localize_individual_images(
                        testImages=(testIm), initial_upsample=ius,
                        showInterpolation=True, verbose=False)
                    pylab.ion()
                    print "(c)ontinue, or enter initial upsample [c]/#:",
                    ius = raw_input()
                    try:
                        ius = float(ius)
                        print "Setting initial upsample to ", ius
                    except ValueError:
                        print "Continuing."
                        break
            else:
                imageNum += 1
        if liveInput: pylab.close('all')
        localizations.close()
        localizations = None ##Prevents a funky error message
        if linkedInput:
            particles.close()
            particles = None
        if liveInput: print "Done inspecting."
        return None

    def link_localizations(
        self,
        linkFilter='default',
        fidFilter='default',
        maxSearchFrames=3,
        saveIndividualInfo=True,
        matchOnlyBrights=False,
        promptForInspection=True
        ):
        """
        After running localize_candidates(), link_localizations()
        attempts to identify which 'birth' and 'death' localizations
        represent the same blinking particle.

        Any particle that satisfies 'fidFilter' is considered a
        'fiducial'. Fiducials do not blink on and off, so we can't do
        background subtraction. Fiducials are often better suited for
        tracking stage drift than characterizing a sample.

        Creates a shelve of linked localization dicts, with the same
        structure as the 'candidates' shelve, except
        'later_image_names' and possibly 'bg_image_names' are filled
        in. This shelve is suitable for feeding back into
        "localize_candidates", hopefully giving improved localization
        since we can average multiple images and subtract backgrounds.

        """
        import os, sys, shelve

        if not os.path.isfile(os.path.join(
            self.imFolder, self.localizations_filename)):
            print "Localizations file '" + self.localizations_filename + "'",
            print " not found."
            print "Locate the file, or run localize_candidates() first."
            return None
        localizations = shelve.open(os.path.join(
            self.imFolder, self.localizations_filename), protocol=2)        
        if os.path.isfile(os.path.join(
            self.imFolder, self.linked_localizations_filename)):
            print "The file '" + self.linked_localizations_filename + "'",
            print "already exists."
            print "Looks like we already linked localizations.",
            print "Overwrite? y/[n]:",
            overWrite = raw_input()
            if overWrite != 'y':
                print "Using old linked localizations file. Be careful!\n"
                return None
            else:
                print "Removing old linked localizations."
                os.remove(os.path.join(
                    self.imFolder, self.linked_localizations_filename))
        linked_localizations = shelve.open(os.path.join(self.imFolder,
            self.linked_localizations_filename), protocol=2)
        """Keep track of how many localized particles. Saves using
        localizations.keys(), which is very slow:"""
        linked_localizations['num'] = 0

        if linkFilter =='default': linkFilter = _linking_filter
        if  fidFilter =='default': fidFilter = _fiducial_filter
        print "Linking localizations..."
        sys.stdout.flush()

        currentImage = 0
        localizations_cache = {}
        (numOrphanBirths, numOrphanDeaths, numOrphanBrights) = (0, 0, 0)
        while currentImage < len(self.images):
            """Get the current-image locs from the cache, load if required:"""
            currentLocs = localizations_cache.pop(
                repr(currentImage), localizations[repr(currentImage)])
            birthLocs = []
            for loc in currentLocs:
                if 'birth_flag' in loc:
                    birthLocs.append([loc])
                elif 'bright_flag' in loc:
                    numOrphanBrights += 1
                elif 'death_flag' in loc:
                    numOrphanDeaths += 1
            matchesList = []
            searchImage = currentImage + 1
            """Search for matches to each birthLoc, retiring birthLocs
            when they get 'cold' or find a death:"""
            while (len(birthLocs) > 0) and (searchImage < len(self.images)):
                searchLocs = localizations_cache.setdefault(
                    repr(searchImage), localizations[repr(searchImage)])
                for matches in birthLocs:
                    matchLocs = [loc for loc in searchLocs if
                                 linkFilter(matches[0], loc)]
                    for loc in matchLocs:
                        searchLocs.remove(loc)    
                    matches.extend(matchLocs)
                orphanBirths = [
                    matches for matches in birthLocs if
                    matches[-1]['image_num'] + maxSearchFrames < searchImage]
                numOrphanBirths += len(orphanBirths)
                doneMatches = [
                    matches for matches in birthLocs if
                    ('death_flag' in matches[-1]) or fidFilter(matches)]
                matchesList.extend(doneMatches) #What order should this be?
                for matches in orphanBirths + doneMatches:
                    birthLocs.remove(matches)
                searchImage += 1
            linkedImageLocs = []
            for matches in matchesList:
                linkedLoc = {'image_name': matches[0]['image_name'],
                             'x_slice': matches[0]['x_slice'],
                             'y_slice': matches[0]['y_slice'],
                             'image_num': matches[0]['image_num']}
                """Which frames get averaged for re-localization:"""
                startFr = matches[0]['image_num'] + 1 #birth frame + 1
                endFr = matches[-1]['image_num']      #death frame
                if matchOnlyBrights:
                    linkedLoc['later_image_names'] = [
                        m['image_name'] for m in matches[1:] if
                        'bright_flag' in m]
                else:
                    linkedLoc['later_image_names'] = self.images[startFr:endFr]
                """Additional stored data:"""
                if saveIndividualInfo:
                    linkedLoc['single_image_localizations'] = matches
                if 'death_flag' in matches[-1]:
                    if (startFr > 1 and endFr < len(self.images)):
                        linkedLoc['bg_image_names'] = [
                            self.images[startFr - 2], self.images[endFr]]
                else: #Don't subtract background from fiducials:
                    linkedLoc['fiducial_flag'] = True
                linkedImageLocs.append(linkedLoc)
            linked_localizations[repr(currentImage)] = linkedImageLocs
            linked_localizations['num'] += len(linkedImageLocs)
            currentImage += 1
            if currentImage%60 == 0:
                linked_localizations.sync()
                sys.stdout.write(
                    "\r%7d remaining localizations"%(
                        localizations['num'] - linked_localizations['num'] -
                        numOrphanBrights - numOrphanBirths - numOrphanDeaths))
                sys.stdout.flush()
        print "\n%i localizations linked into %i logical particles."%(
            localizations['num'], linked_localizations['num'])
        print "%i orphaned bright localizations"%(numOrphanBrights)
        print "%i orphaned birth localizations"%(numOrphanBirths)
        print "%i orphaned death localizations"%(numOrphanDeaths)
        localizations.close()
        linked_localizations.close()
        localizations = None ##Prevents a funky error message, sometimes
        linked_localizations = None
        if promptForInspection:
            print "Inspect linking? [y]/n:",
            inspectLinking = raw_input()
            if inspectLinking != 'n':
                self.inspect_linking()
        return None

    def inspect_linking(self):
        import os, shelve, pprint
        import pylab

        if not os.path.isfile(os.path.join(
            self.imFolder, self.linked_localizations_filename)):
            print "Localizations file '" + self.linked_localizations_filename,
            print "' not found."
            print "Locate the file, or run link_localizations() first."
            return None
        linked_localizations = shelve.open(os.path.join(
            self.imFolder, self.linked_localizations_filename), protocol=2)
        imageNum = 0
        particleNum = 0
        maxImages = len(self.images)
        pylab.close('all')
        pylab.figure()
        while True:
            """If imageNum, particleNum exists, show. If not, find the
            next existing localization. If nothing, show nothing."""
            while True:
                if imageNum < maxImages: ##Assumes sequentially numbered images
                    locList = linked_localizations[repr(imageNum)]
                    if len(locList) > particleNum:
                        loc = locList[particleNum]
                        break
                    else:
                        imageNum += 1
                        particleNum = 0
                else: ##Find the last localization
                    imageNum = maxImages - 1
                    while True:
                        locList = linked_localizations[repr(imageNum)]
                        if len(locList) > 0:
                            break
                        else:
                            imageNum -= 1
                    loc = locList[-1]
                    print "Last localization: image %i, localization %i"%(
                        imageNum, len(locList) - 1)
                    break
            pylab.suptitle(" "*70 + "Drawing...")
            pylab.ioff()
            self.show_linked_localization(
                loc, imageNum, particleNum, newFig=False)
            pylab.ion()
            if 'single_image_localizations' in loc:
                for i, lo in enumerate(loc['single_image_localizations']):
                    print "Im %i: x=%5.2f y=%5.2f z=%5.2f qual=%5.2f"%(
                        i+1, lo['x'], lo['y'], lo['z'], lo['qual']),
                    print "in frame %i"%(lo['image_num'])
            print "\n(c)ontinue, (j)ump images, (p)rint details,",
            print "or (f)inished inspecting? [c]/j/p/f:",
            cmd = raw_input()
            if cmd == 'f':
                break
            elif cmd == 'j':
                while True:
                    print "New image number:",
                    imageNum = raw_input()
                    try:
                        imageNum = int(imageNum)
                        break
                    except ValueError:
                        print "Type an integer, hit return."
                while True:
                    print "New particle number [0]:",
                    particleNum = raw_input()
                    if particleNum == '':
                        particleNum = 0
                        break
                    else:
                        try:
                            particleNum = int(particleNum)
                            break
                        except ValueError:
                            print "Type an integer, hit return."
            elif cmd == 'p':
                print ''
                pprint.pprint(loc)
                print ''
            else:
                particleNum += 1
        pylab.close('all')
        linked_localizations.close()
        linked_localizations = None ##Prevents a funky error message
        print "Done inspecting."
        return None

    def show_linked_localization(
        self, loc, imageNum, particleNum, newFig=True):
        import scipy, pylab

        locImage = self.im_load(
            loc['image_name'])[loc['x_slice'], loc['y_slice']]
        locImages = [1.0 * locImage]
        bgImage = scipy.zeros(locImage.shape)
        bgImages = []
        if 'later_image_names' in loc:
            numImages = min(len(loc['later_image_names']) + 1, 15)
            for i, im in enumerate(loc['later_image_names']):
                newImage = self.im_load(im)[loc['x_slice'], loc['y_slice']]
                locImage += newImage
                if len(locImages) < numImages:
                    locImages.append(newImage[:,:])
            locImage = locImage * (1.0/(1. + len(loc['later_image_names'])))
        avgCounts = locImage.mean()
        if 'bg_image_names' in loc:
            numBG = min(len(loc['bg_image_names']), 2)
            for i, im in enumerate(loc['bg_image_names']):
                newImage = self.im_load(im)[loc['x_slice'], loc['y_slice']]
                bgImage += newImage
                if len(bgImages) < numBG:
                    bgImages.append(1.0 * newImage)
            bgImage = bgImage * 1.0/(len(loc['bg_image_names']))
            avgCountsAboveBG = (locImage - bgImage).mean()
        if newFig: pylab.figure()
        pylab.clf()
        pylab.suptitle("Linked images for particle %i in frame %i"%(
            particleNum, imageNum))
        subplotNum = 1
        for im in [(locImage-bgImage),locImage,bgImage] + bgImages + locImages:
            pylab.subplot(4,5,subplotNum)
            pylab.imshow(im, interpolation='nearest', cmap=pylab.cm.gray)
            if subplotNum == 1:
                titleStr = 'Signal'
            elif subplotNum == 2:
                titleStr = 'Im Avg'
            elif subplotNum == 3:
                titleStr = 'BG Avg'
            elif subplotNum <= 3 + len(bgImages):
                titleStr = 'BG#%i'%(subplotNum - 3)
            else:
                titleStr = 'Im#%i'%(subplotNum - 3 - len(bgImages))
            pylab.title(titleStr)
            pylab.xticks([])
            pylab.yticks([])
            subplotNum += 1
        pylab.gcf().show()
        return None

    def select_fiducials(self):
        """
        The user picks out fiducials from the localization data.

        The user's descriptions of the fiducials are used to construct
        filtering functions. A python module is constructed which
        contains these function definitions, which can be imported
        later anytime fiducial filtering is needed.

        I know code-writing-code is an abomination, but the output
        module has the nice side-effect of giving a non-programming
        user an example of how to construct filters.

        """
        import os
        import pylab

        print "Selecting fiducials..."
        if not hasattr(self, 'fiducial_filter_filename'):
            raise UserWarning(
                "fiducial_filter_filename is not set. Set it first!")
        if os.path.isfile(self.fiducial_filter_filename):
            print "The file '" + self.fiducial_filter_filename + "'",
            print "already exists."
            print "Looks like we already selected fiducials. Overwrite? y/[n]:",
            overWrite = raw_input()
            if overWrite != 'y':
                print "Using old fiducial selections file. Be careful!\n"
                return None
            else:
                print "Removing old fiducial selections."
        fidFile = open(self.fiducial_filter_filename, 'w')
        fidFile.write("names = []")
        fidFile.close()
        fid_filters = __import__(
            os.path.splitext(self.fiducial_filter_filename)[0])
        reload(fid_filters) ##To prevent funkiness from previous runs
        bytecodeFilename = self.fiducial_filter_filename + 'c'
        if os.path.exists(bytecodeFilename): #Clean up a little
            os.remove(bytecodeFilename)
        filterDefString = ""
        filterListString = ""
        while True:
            print "\n(a)dd fiducial, (c)lear fiducials, (p)lot localizations,",
            print "(s)how images, or\n(f)inished? a/c/p/s/f:",
            userResponse = raw_input()
            if userResponse == 'a' or userResponse == 'c':
                """Edit the fiducial filters"""
                if userResponse == 'a':
                    print "Choose filter values:"
                    filterVals={}
                    keyList = ['xMax', 'xMin', 'yMax', 'yMin', 'zMax', 'zMin',
                                'correlationMin']
                    if self.im_z_positions is not None:
                        keyList.extend(('piezoMax', 'piezoMin'))
                    for key in keyList:
                        while True:
                            print key + ":",
                            val = raw_input()
                            try:
                                val = float(val)
                                break
                            except ValueError:
                                print "Type a number, hit return."
                        filterVals[key] = val
                        print key + "=", val
                    for a in ('x', 'y', 'z'): #Sanity checking:
                        if filterVals[a + 'Max'] < filterVals[a + 'Min']:
                            print "\nCareful! " + a + "Max < " + a + "Min.\n"
                    funcNum = len(fid_filters.names)
                    filterVals['funcNum'] = funcNum
                    filterDefString += _localization_filter_string(**filterVals)
                    filterListString += 'locFilter%i, \n'%(funcNum)
                    print "Fiducial added."
                elif userResponse == 'c':
                    filterDefString = ""
                    filterListString = ""
                    print "Fiducials cleared."
                fidFile = open(self.fiducial_filter_filename, 'w')
                fidFile.write(
                    filterDefString + 'names = [\r\n' + filterListString + ']' +
                    '\r\n\r\n"""\r\nPossibly useful keys for custom filters:' +
                    '\r\nx, y, z, qual, image_num, z_piezo\r\n' +
                    'bright_flag, birth_flag, death_flag, edge_flag\r\n' +
                    '"""')
                fidFile.close()
                reload(fid_filters)
                bytecodeFilename = self.fiducial_filter_filename + 'c'
                if os.path.exists(bytecodeFilename): #Clean up a little
                    os.remove(bytecodeFilename)
            elif userResponse == 'p':
                reload(fid_filters)
                bytecodeFilename = self.fiducial_filter_filename + 'c'
                if os.path.exists(bytecodeFilename): #Clean up a little
                    os.remove(bytecodeFilename)
                def unfiltered(loc):
                    for filt in fid_filters.names:
                        if filt(loc):
                            return False
                    return True
                pylab.close('all')
                self.plot_xyz_localizations(
                    locFilters=(fid_filters.names + [unfiltered]))
            elif userResponse == 's':
                pylab.clf()
                pylab.suptitle('Type an image number')
                while True:
                    print "Type an image number, or [e]xit showing images:",
                    imageNum = raw_input()
                    try:
                        imageNum = int(imageNum)
                    except ValueError:
                        break
                    if imageNum >= len(self.images):
                        print "There are only %i images"%(len(self.images))
                        imageNum = len(self.images) - 1
                    pylab.ioff()
                    pylab.clf()
                    pylab.imshow(self.im_load(self.images[imageNum]),
                                 interpolation='nearest', cmap=pylab.cm.gray)
                    pylab.suptitle('Image number %i'%(imageNum))
                    pylab.gcf().show()
                    pylab.ion()
                print "Done showing images."
                pylab.close('all')
            elif userResponse == 'f':
                break
        pylab.close('all')
        print "Done selecting fiducials."
        print "Determine sample drift? [y]/n:",
        driftCorrect = raw_input()
        if driftCorrect != 'n':
            self.fiducials_to_drift()
        return None

    def fiducials_to_drift(self, smoothing_sigma=20, zPiezoCorrection=True):
        """
        Use the fiducial filters constructed by 'select_fiducials()'
        to calculate sample drift over an acquisition.

        Choosing an appropriate 'smoothing_sigma' is important for
        good localizations. Don't be afraid to oversmooth; if the
        sample is jumping around rapidly, the data is probably lousy
        anyway.
        
        """
        import os
        import scipy, pylab
        from scipy.ndimage import gaussian_filter1d

        if not hasattr(self, 'fiducial_filter_filename'):
            raise UserWarning('fiducial_filter_filename is not set.')
        if not os.path.isfile(self.fiducial_filter_filename):
            raise UserWarning(
                "Fiducial selection file '" +
                self.fiducial_filter_filename + "'" + " not found.\n" +
                "Locate the file, or run select_fiducials() first.")
            return None
        else:
            fid_filters = __import__(
                os.path.splitext(self.fiducial_filter_filename)[0])
            reload(fid_filters)
            bytecodeFilename = self.fiducial_filter_filename + 'c'
            if os.path.exists(bytecodeFilename): #Clean up a little
                os.remove(bytecodeFilename)
        print "Loading drift data..."
        if zPiezoCorrection and self.im_z_positions is None:
            zPiezoCorrection = False
            print "'im_z_positions' is not specified. Be careful!"
        print "Z-piezo correction: ", zPiezoCorrection
        (figs, x, y, z, qual, image_num) = self.plot_xyz_localizations(
            locFilters=fid_filters.names, zPiezoCorrection=zPiezoCorrection)
        xyz_nodupes = []
        (x_nd, y_nd, z_nd, image_num_nd) = ([],[],[],[])
        pylab.ioff()
        for i in range(len(figs)):
            """A fiducial can only localize once per image for drift
            correction, so we remove lower-quality duplicates:"""
            xyz_nodupes.append({}) #image_num keys, (x,y,z,qual) tuple values
            for a in (x_nd, y_nd, z_nd, image_num_nd):
                a.append([])
            for j, num in enumerate(image_num[i]):
                if num in xyz_nodupes[-1]:
                    existing_qual = xyz_nodupes[-1][num][3]
                    if qual[i][j] < existing_qual:
                        continue #Don't include this low-quality localization
                xyz_nodupes[-1][num] = (
                    x[i][j], y[i][j], z[i][j], qual[i][j])
            """Convert back to lists:"""
            for k in sorted(xyz_nodupes[-1].keys()):
                xyz = xyz_nodupes[-1][k]
                x_nd[-1].append(xyz[0])
                y_nd[-1].append(xyz[1])
                z_nd[-1].append(xyz[2])
                image_num_nd[-1].append(k)
            """Interpolate and smooth the drift coordinates:"""
            ims = range(len(self.images))
            for a in (x_nd, y_nd, z_nd):
                a[-1] = gaussian_filter1d(scipy.interp(
                    ims, image_num_nd[-1], a[-1]), sigma=smoothing_sigma)
            xAx, yAx, zAx, qualAx  = figs[i].get_axes()
            xAx.plot(ims, x_nd[-1])
            yAx.plot(ims, y_nd[-1])
            zAx.plot(ims, z_nd[-1])
        pylab.ion()
        pylab.ioff()
        pylab.figure()
        pylab.suptitle(r"XYZ drift data, smoothed with $\sigma$=%0.2f pixels"%(
            smoothing_sigma))
        pylab.subplot(3, 1, 1)
        for i in range(len(figs)):
            pylab.plot(ims, x_nd[i] - x_nd[i][0],
                       label = "Fiducial %i"%(i))
        pylab.grid()
        pylab.legend()
        pylab.ylabel("x-pixel drift")
        pylab.subplot(3, 1, 2)
        for i in range(len(figs)):
            pylab.plot(ims, y_nd[i] - y_nd[i][0])
        pylab.grid()
        pylab.ylabel("y-pixel drift")
        pylab.subplot(3, 1, 3)
        for i in range(len(figs)):
            pylab.plot(ims, z_nd[i] - z_nd[i][0])
        pylab.grid()
        pylab.ylabel("z-pixel drift")
        pylab.xlabel("Image number")
        pylab.gcf().show()
        pylab.ion()
        print ("\n(c)ontinue, (r)esmooth, (z)-piezo correction," +
               " or (e)dit filter? [c]/r/z/e:"),
        reSmooth = raw_input()
        if reSmooth == 'e':
            print "Edit", self.fiducial_filter_filename
            print "When finished editing, hit return:",
            raw_input()
        if reSmooth == 'r' or reSmooth == 'e' or reSmooth == 'z':
            while True:
                print "New smoothing sigma: [%0.2f]"%(smoothing_sigma),
                smoothingSigma = raw_input()
                if smoothingSigma == '':
                    smoothingSigma = smoothing_sigma
                try:
                    smoothingSigma = float(smoothingSigma)
                    break
                except ValueError:
                    print "Type a number, hit return."
            pylab.close('all')
            if reSmooth == 'z':
                if self.im_z_positions is None:
                    print "'im_z_positions' is not set. No z-piezo correction."
                    zPiezoCorrection = False
                else:
                    zPiezoCorrection = not zPiezoCorrection
                    print "\nz-piezo correction:", zPiezoCorrection, "\n"
            return self.fiducials_to_drift(smoothing_sigma=smoothingSigma,
                                           zPiezoCorrection=zPiezoCorrection)

        while True:
            print "What fiducial should we use for drift tracking?:",
            whichFid = raw_input()
            try:
                whichFid = int(whichFid)
            except ValueError:
                print "Type an integer, hit return."
            if whichFid in range(len(figs)):
                print "Using fiducial %i"%(whichFid)
                break
            else:
                print "Valid fiducials are", range(len(figs))
        pylab.close('all')
        """Subtract off the mean position of the fiducial to give drift:"""
        self.drift = {
            'x': x_nd[whichFid] - x_nd[whichFid][0],
            'y': y_nd[whichFid] - y_nd[whichFid][0],
            'z': z_nd[whichFid] - z_nd[whichFid][0],
            'initial_xyz': (
                x_nd[whichFid][0], y_nd[whichFid][0], z_nd[whichFid][0])}
        print "Drift correction determined."
        self.save(overwriteWarning=False)
        return None

    def localize_individual_images(
        self, testImages, initial_upsample=0.35,
        showInterpolation=False, verbose=True):
        """
        Localizes a small number of images, given as 'testImages', a
        3d scipy array.

        testImages[:,:,n] is the nth image, and must
        have the same shape as self.calibration[:,:,0].

        """
        import os, shelve
        import scipy

        testImages = scipy.atleast_3d(scipy.asarray(testImages))
        images = range(testImages.shape[2])
        def imLoad(im):
            return testImages[:,:,int(im)]
        """Construct an artificial version of 'candidates':"""
        candName = "DELETEME_" + os.path.split(self.candidates_filename)[1]
        locName = "DELETEME_" + os.path.split(self.localizations_filename)[1]
        checkCandidates = shelve.open(os.path.join(
            self.imFolder, candName), protocol=2)
        for image_num in images:
            checkCandidates[repr(image_num)] = [{
                'image_name': image_num,
                'x_slice': slice(0, self.calibration.shape[0]),
                'y_slice': slice(0, self.calibration.shape[1]),
                'image_num': image_num}]
        checkCandidates['num'] = len(images)
        checkCandidates.close()
        temp_palm = Palm_3d(
            images=images, im_format='custom', custom_im_load=imLoad,
            calibration=self.calibration, imFolder=self.imFolder,
            candidates_filename=candName, localizations_filename=locName)
        """Use our correlation tool to correlate every test image.
        There's some unneccesary overhead here, but it beats
        duplicating code:"""
        temp_palm.localize_candidates(
            saveCorrelations=True, initial_upsample=initial_upsample,
            verbose=verbose, promptForInspection=False,
            showInterpolation=showInterpolation, retainEdgeFlags=True)
        """Now we've got a correlation stack for every input image.
        For calibration checking, we want to turn this into an MxM
        matrix of maximum correlations, where M is the number of
        calibration slices, and 'maximum' is taken across
        xy-shifts."""
        checkLocalizations = shelve.open(os.path.join(
            self.imFolder, locName), protocol=2)
        calibrationCrossterms = scipy.zeros(
            (len(images),
             checkLocalizations['0'][0]['correlations'][0].shape[2]))
        for i in images:
            calibrationCrossterms[i,:] = (
                checkLocalizations[repr(i)][0]['correlations'][0].max(0).max(0))
        checkLocalizations.close()
        os.remove(os.path.join(self.imFolder, candName))
        os.remove(os.path.join(self.imFolder, locName))
        return calibrationCrossterms

    def histogram_3d(
        self,
        xBins='pixels',
        yBins='pixels',
        zBins='pixels',
        nm_per_pixel=(None, None, None),
        locFilters=None,
        cornersOrCenters='centers',
        driftCorrection=True,
        zPiezoCorrection=True,
        persistent=True,
        overwriteWarning=True,
        memoryWarning=True,
        recomputeHistogram=None,
        initialOffset=(0,0,0),
        initialHistograms=None,
        linkedInput=False):
        """
        Construct a histogram of localizations vs. position.

        zBins should be a list of floating point numbers giving the
        positions of bin edges, like with matplotlib's 'hist'
        function. Be aware that lots of bins might take LOTS of
        memory. 'xBins' and 'yBins' can also be specified similarly,
        and default to the pixel edges of the original images.

        'locFilters' is a list of localization filtering functions,
        similar to the functions generated by 'select_fiducials()'.
        One histogram will be generated for each filter, containing
        only localizations for which the filter returns 'True'.

        'nm_per_pixel' is a 3-tuple giving the spatial calibration of
        the image in nanometers per pixel. Beware that not all image
        display programs agree on which way is 'x' and which way is
        'y'. The xyz convention here is the same as our localizations
        dictionaries.

        'cornersOrCenters': x and y localizations are stored as the
        coordinates of the upper left corner of a calibration image
        which would best match the image molecule. Because the precise
        position of the calibration fiducial within the calibration
        image is unknown, we measure only relative localizations, and
        the global position of the localizations is unspecified. If
        you want to plot these 'corners' localizations, use:
        
        cornersOrCenters='corners'.

        If you want to plot the positions of the approximate center of
        the calibration window, use:

        cornersOrCenters='centers'

        'driftCorrection' and 'piezoCorrection': If the sample drift
        attribute 'drift' is set, you can use 'driftCorrection=True'
        to subtract known xyz sample drift from each measured
        localization. If the z-piezo position vs. frame number
        attribute 'im_z_positions' is set, use 'piezoCorrection=True'
        to subtract known z-piezo positions from each measured
        localization.

        Computing a histogram can be time-consuming, but the result
        may not be too large to store and load. Use 'persistent=True'
        to save recomputation time, and 'persistent=False' to save
        disk space.

        The user may want to combine histograms from two different
        datasets. It is important to construct these histograms using
        the same bins and the same reference postion. Set
        'initialOffset' to align the current dataset's histogram with
        another dataset's histogram.

        """
        inputArguments = locals()
        inputArguments.pop('self')
        inputArguments['locFilters']='unknown' ##Simplifies persistence
        import os, sys, cPickle, shelve
        import scipy

        histFile = os.path.join(self.imFolder, 'stored_histograms.pkl')
        if persistent and os.path.exists(histFile):
            print "Loading old stored histograms..."
            try:
                stored_histograms = cPickle.load(open(histFile, 'rb'))
            except IOError:
                print "Load failed. Recomputing."
                stored_histogram = {'input_arguments': None}
            if (stored_histograms['input_arguments'] == repr(inputArguments)):
                if recomputeHistogram is None or recomputeHistogram =='ask':
                    print "Looks like we already computed a similar histogram."
                    print "Recompute? y/[n]:",
                    recomputeHistogram = raw_input()
                if recomputeHistogram == 'y':
                    print "Recomputing histogram."
                    os.remove(histFile)
                    stored_histograms = None
                else:
                    print "Using previously computed histogram."
                    return stored_histograms['histograms']

        if (xBins == 'pixels') or (yBins == 'pixels'):
            xyShape = self.im_load(self.images[0]).shape
            if xBins == 'pixels':
                xBins = range(xyShape[0] + 1)
            if yBins == 'pixels':
                yBins = range(xyShape[1] + 1)
        if zBins == 'pixels':
            zBins = range(self.calibration.shape[2] + 1)
        if locFilters is None:
            def nullFilter(loc):
                return True
            locFilters = [nullFilter]
        (xOffset, yOffset, zOffset) = initialOffset
        if cornersOrCenters == 'centers':
            xOffset += 0.5 * self.calibration.shape[0]
            yOffset += 0.5 * self.calibration.shape[1]
        numMiB = ( (1.0/(2**20)) * ## 4 bytes per bin if dtype = uint32
            len(locFilters)*(len(xBins)-1)*(len(yBins)-1)*(len(zBins)-1) * 4)
        showProgress=False
        if numMiB > 10.:
            showProgress=True
            if memoryWarning:
                print "Careful, histogram_3d will use at least",
                print "%0.2f MiB. Abort? y/[n]:"%(numMiB),
                abortBigHist = raw_input()
                if abortBigHist == 'y':
                    print "*\n"*5 + "Histogram aborted.\n" + "*\n"*5
                    return None
        if initialHistograms is None:
            histograms = []
            for filt in locFilters:
                histograms.append(scipy.zeros(
                    (len(xBins)-1, len(yBins)-1, len(zBins)-1),
                    dtype=scipy.uint32))
        else:
            histograms = initialHistograms
        if linkedInput:
            localizations_filename = self.particles_filename
        else:
            localizations_filename = self.localizations_filename
        if not os.path.isfile(os.path.join(
            self.imFolder, localizations_filename)):
            print ("Localizations file '" + localizations_filename +
                   "' not found.")
            print "Locate the file, or run 'localize_candidates()' first."
            return None
        localizations = shelve.open(os.path.join(
            self.imFolder, localizations_filename), protocol=2)
        xyz = []
        for locFilter in locFilters:
            xyz.append(([], [], []))
        for i in range(len(self.images)):
            locList = localizations[repr(i)]
            for loc in locList:
                if showProgress and (i%50 == 0 or i == len(self.images) - 1):
                    sys.stdout.write("\rImages processed:%7d/%d"%(
                        i+1, len(self.images)))
                    sys.stdout.flush()
                """Accumulate a set of localizations:"""
                for f, locFilter in enumerate(locFilters):
                    if locFilter(loc):
                        (x, y, z) = xyz[f]
                        x.append(loc['x'] + xOffset)
                        y.append(loc['y'] + yOffset)
                        z.append(loc['z'] + zOffset)
                        if zPiezoCorrection:
                            if hasattr(self, 'im_z_positions'):
                                z[-1] -= self.im_z_positions[loc['image_name']]
                            else:
                                raise UserWarning("'im_z_positions' not set," +
                                                  "so no z-piezo correction.")
                        if driftCorrection:
                            if hasattr(self, 'drift'):
                                x[-1] -= self.drift['x'][loc['image_num']]
                                y[-1] -= self.drift['y'][loc['image_num']]
                                z[-1] -= self.drift['z'][loc['image_num']]
                            else:
                                raise UserWarning(
                                    "Drift correction not set. " +
                                    "Run fiducials_to_drift() first.")
            if len(xyz[0][0]) > 5e4 or i == len(self.images)-1:
                if showProgress:
                    sys.stdout.write("\rAppending to histogram..." + ' '*20)
                    sys.stdout.flush()
                for f in range(len(locFilters)):
                    if len(xyz[f][0]) == 0: continue
                    H, edges = scipy.histogramdd(
                        sample=xyz[f], bins=(xBins, yBins, zBins))
                    histograms[f] += H.astype(scipy.uint32)
                    xyz[f] = ([], [], [])
        localizations.close()
        localizations = None
        if persistent:
            print "\nStoring computed histogram..."
            stored_histograms = {
                'input_arguments': repr(inputArguments),
                'histograms': histograms,
                'bins': (xBins, yBins, zBins),
                'nm_per_pixel' : nm_per_pixel}
            cPickle.dump(stored_histograms, open(histFile, 'wb'), protocol=2)
        return histograms

    def plot_xyz_localizations(
        self,
        locFilters=None,
        zPiezoCorrection=False,
        driftCorrection=False,
        decimationFactor=1,
        linkedInput=False,
        zSubtraction=None):
        """ Plots particle xyz localization and correlation strength
        vs. image number.

        'locFilters' is a list of functions. Each function takes a
        localization dict and returns True or False. This allows the
        user to pick out particular localizations to plot. The results
        from each function in locFilters will be plotted in a separate
        window.

        'zPiezoCorrection' is True or False, specifying whether or not to
        subtract known z-positions from each image.

        This function requires the 'image_num' part of each
        localization dict is defined.

        'zSubtraction' is deprecated. Use "zPiezoCorrection" instead.

        """
        import os, sys, shelve
        import scipy, pylab

        if zSubtraction is not None:
            print "\nDeprecation warning- zSubtraction is deprecated."
            print "Use zPiezoCorrection instead.\n"

        if locFilters is None:
            def nullFilter(loc):
                return True
            locFilters = [nullFilter]
        (x, y, z, qual, image_num) = [[[] for i in range(len(locFilters))]
                                      for j in range(5)]
        if linkedInput:
            localizations_filename = self.particles_filename
        else:
            localizations_filename = self.localizations_filename
        if not os.path.isfile(os.path.join(
            self.imFolder, localizations_filename)):
            print ("Localizations file '" + localizations_filename +
                   "' not found.")
            print "Locate the file, or run 'localize_candidates()' first."
            return None
        localizations = shelve.open(os.path.join(
            self.imFolder, localizations_filename), protocol=2)
        showProgress = False
        while True:
            imRange = range(0, len(self.images), decimationFactor)
            estimatedLocNum = (localizations['num'] *
                               len(imRange) * 1./len(self.images))
            if estimatedLocNum < 1e4:
                break
            print "%i images will be loaded, containing about %i points."%(
                len(imRange), int(estimatedLocNum)),
            print "Decimate? [y]/n:",
            decimateNow = raw_input()
            if decimateNow != 'n':
                while True:
                    print "Decimation factor:",
                    newFactor = raw_input()
                    try:
                        decimationFactor = int(newFactor)
                        break
                    except ValueError:
                        print "Type an integer, hit return"
                print "Decimation factor set to", decimationFactor
            else:
                showProgress = True
                break
        for i in imRange:
            if showProgress and (i%50 == 0 or i == imRange[-1]):
                sys.stdout.write(
                    "\rImages processed:%7d/%d"%(
                        i+1, len(self.images)))
                sys.stdout.flush()
            locList = localizations[repr(i)]
            for loc in locList:
                for f, locFilter in enumerate(locFilters):
                    if locFilter(loc):
                        x[f].append(loc['x'])
                        y[f].append(loc['y'])
                        z[f].append(loc['z'])
                        qual[f].append(loc['qual'])
                        image_num[f].append(loc['image_num'])
                        if zPiezoCorrection:
                            if hasattr(self, 'im_z_positions'):
                                z[f][-1] -= self.im_z_positions[
                                    loc['image_name']]
                            else:
                                raise UserWarning("'im_z_positions' not set," +
                                                  "so no z-piezo correction.")
                        if driftCorrection:
                            if hasattr(self, 'drift'):
                                x[f][-1] -= self.drift['x'][loc['image_num']]
                                y[f][-1] -= self.drift['y'][loc['image_num']]
                                z[f][-1] -= self.drift['z'][loc['image_num']]
                            else:
                                raise UserWarning(
                                    "Drift correction not set. " +
                                    "Run fiducials_to_drift() first.")
        localizations.close()
        localizations=None ##To suppress a strange error message
        figs = []
        for f in range(len(locFilters)):
            figs.append(pylab.figure())
            pylab.suptitle("Drawing...")
            pylab.ioff()
            pylab.clf()
            titleStr = ('X, Y, and Z localization vs. image number. ' +
                        '%i localizations for filter %i.'%(len(x[f]), f))
            if zPiezoCorrection:
                titleStr += " Z-piezo corrected."
            if driftCorrection:
                titleStr += " Drift corrected."
            pylab.suptitle(titleStr)                
            for whichSub, (a, myTitle) in enumerate(
                ([x[f], 'x-pixel location'],
                 [y[f], 'y-pixel location'],
                 [z[f], 'z-pixel location'],
                 [qual[f], 'Correlation strength'])):
                if len(image_num[f]) == 0:
                    print "No localizations passed filter %i"%(f)
                    break
                pylab.subplot(4, 1, whichSub+1)
                pylab.hold('on')
                pylab.scatter(
                    image_num[f], a, c=qual[f], s=10, edgecolors='none')
                pylab.ylabel(myTitle)
                pylab.axis('tight')
                pylab.grid()
            pylab.xlabel('Image number (i)')
        for fig in figs:
            fig.show()
        pylab.ion()
        return (figs, x, y, z, qual, image_num)

    def diffraction_limited_image(
        self, images='all', showImage=False, persistent=True):
        import sys
        import scipy, pylab

        loadImages=True
        if persistent and hasattr(self, 'average_image'):
            print "Diffraction-limited image already computed."
            print "Recompute? y/[n]:",
            reCompute = raw_input()
            if reCompute != 'y':
                print "Using previously computed image."
                diffractionLimitedImage = 1.0*self.average_image
                loadImages = False
        if loadImages:
            print "Computing diffraction-limited image."
            if images == 'all':
                images = self.images
            diffractionLimitedImage = scipy.zeros(self.im_load(images[0]).shape)
            for i, im in enumerate(images):
                if i%50 == 0 or i+1 == len(self.images):
                    sys.stdout.write("\rImages processed:%7d/%d"%(
                        i+1, len(self.images)))
                    sys.stdout.flush()
                diffractionLimitedImage += self.im_load(im)
            diffractionLimitedImage /= len(images)
            print ""
            if persistent:
                self.average_image = diffractionLimitedImage
                self.save()
        if showImage:
            pylab.figure()
            pylab.imshow(diffractionLimitedImage, interpolation='nearest')
            pylab.gcf().show()
        return diffractionLimitedImage

def combine_palm_histograms(
    palmAcquisitions,
    locFilters=None,
    xBins='pixels',
    yBins='pixels',
    zBins='pixels',
    nm_per_pixel=(None, None, None),
    cornersOrCenters='centers',
    driftCorrection=True,
    zPiezoCorrection=True,
    persistent=True,
    persistentWarning=True,
    recomputeHistograms=False,
    promptForDisplay=True,
    linkedInput=False):

    """
    Constructs localization histograms from multiple saved Palm_3d
    acquisitions.
    
    'palmAcquisitions' is a list of saved Palm_3d class instances. For
    example:
        palmAcquisitions = ['0_palm_acquistion.pkl', '1_palm_acquistion.pkl']

    If 'driftCorrection==True', all the saved acquisitions must have
    the 'drift' attribute and must use the same fiducial for drift
    tracking.
    
    'locFilters' is a list of lists of localization filters, one list
    for each palm acquisition in 'palmAcquisitions'. Each list in
    locFilters must be the same length.

    See the 'Palm_3d().histogram_3d' docstring for more information.

    """
    import scipy, numpy, pylab

    print "\nConstructing combined 3D histogram."

    if persistent and persistentWarning:
        print "Persistent mode is on- stored histograms will be overwritten."
        print "Continue? [y]/n:",
        keepGoing = raw_input()
        if keepGoing == 'n':
            raise UserWarning("combine_palm_histograms() aborted.")
    if locFilters is None:
        locFilters = [None] * len(palmAcquisitions)

    for a, acq in enumerate(palmAcquisitions):
        print "\nProcessing acquisition %i: '"%(a), acq, "'."
        data = load_palm(acq)
        if driftCorrection:
            if not hasattr(data, 'drift'):
                raise UserWarning("Drift correction not set." +
                                  " Run fiducials_to_drift() first.")
        if a == 0:
            histograms = None
            initialFiducialXYZ = scipy.array(data.drift['initial_xyz'])
        initialOffset = initialFiducialXYZ - data.drift['initial_xyz']
        print "Initial offset:", initialOffset
        memoryWarning = True
        if a > 0:
            memoryWarning = False
        histograms = data.histogram_3d(
            locFilters=locFilters[a], initialHistograms=histograms,
            xBins=xBins, yBins=yBins, zBins=zBins, persistent=persistent,
            nm_per_pixel=nm_per_pixel, overwriteWarning=False,
            memoryWarning=memoryWarning, recomputeHistogram=recomputeHistograms,
            driftCorrection=driftCorrection, zPiezoCorrection=zPiezoCorrection,
            cornersOrCenters=cornersOrCenters, initialOffset=initialOffset,
            linkedInput=linkedInput)
        if a == 0 and persistent:
            import cPickle
            histFile = os.path.join(data.imFolder, 'stored_histograms.pkl')
            stored_histograms = cPickle.load(open(histFile, 'rb'))
            (xBins, yBins, zBins) = stored_histograms['bins']
    displayHistogram = 'n'
    if promptForDisplay:
        print "\nDisplay 3D histogram? [y]/n:",
        displayHistogram = raw_input()
        if displayHistogram != 'n':
            pylab.figure()
            whichFilter = 0
            whichProjection = 'sum'
            (vmin, vmax) = (None, None)
            logScale = True
            xyzSlice = (
                slice(0, histograms[0].shape[0]),
                slice(0, histograms[0].shape[1]),
                slice(0, histograms[0].shape[2]))
    while displayHistogram != 'n':
        pylab.ioff()
        pylab.clf()
        pylab.suptitle(
            "Filter %i, Projection:%s, Brightness range:%s,%s, Log scale: %s"%(
            whichFilter, whichProjection, vmin, vmax, logScale))
        myHist = histograms[whichFilter][xyzSlice[0], xyzSlice[1], xyzSlice[2]]
        if whichProjection == 'sum':
            myProjection = lambda x: scipy.sum(x, axis=2)
        elif whichProjection == 'max':
            myProjection = lambda x: numpy.max(x, axis=2)
        if logScale:
            myScale = lambda x: scipy.log10(x + 1)
        else:
            myScale = lambda x: x
        """Top view, Z-integrated"""
        pylab.subplot(2, 2, 2)
        plotMe = myScale(myProjection(myHist))
        extent = (xyzSlice[1].start, xyzSlice[1].stop,
                  xyzSlice[0].stop, xyzSlice[0].start)
        aspect = (
            ((xBins[-1] - xBins[0]) * 1.0 / (len(xBins) - 1)) * 1.0 /
            ((yBins[-1] - yBins[0]) * 1.0 / (len(yBins) - 1)))
        if nm_per_pixel != (None, None, None):
            aspect *= nm_per_pixel[0] * 1.0/nm_per_pixel[1]
        pylab.imshow(
            plotMe, interpolation='nearest', cmap=pylab.cm.hot,
            vmin=vmin, vmax=vmax, aspect=aspect, extent=extent)
        pylab.title("Top view")
        pylab.colorbar(pad=0., shrink=0.6)
        scalebarLen = (1000. * (len(yBins) - 1) / #1 micron, in bins
                       (nm_per_pixel[1]*(yBins[-1] - yBins[0])))
        pylab.arrow(extent[0] + (extent[1] - extent[0])/20,
                    extent[3] + (extent[2] - extent[3])/20, scalebarLen, 0,
                    linewidth=4, ec='w', fc='w')
        pylab.text(extent[0] + scalebarLen/2,
                   0.8*extent[3] + 0.2*extent[2], r"1 $\mu$m", color='w')
        """Front view, X-integrated"""
        pylab.subplot(2, 2, 4)
        plotMe = myScale(myProjection(scipy.transpose(myHist, (2,1,0))))
        aspect = (
            ((zBins[-1] - zBins[0]) * 1.0 / (len(zBins) - 1)) * 1.0 /
            ((yBins[-1] - yBins[0]) * 1.0 / (len(yBins) - 1)))
        if nm_per_pixel != (None, None, None):
            aspect *= nm_per_pixel[2] * 1.0/nm_per_pixel[1]
        extent = (xyzSlice[1].start, xyzSlice[1].stop,
                  xyzSlice[2].stop, xyzSlice[2].start)
        pylab.imshow(
            plotMe, interpolation='nearest', cmap=pylab.cm.hot,
            vmin=vmin, vmax=vmax, aspect=aspect, extent=extent)
        pylab.title("Front view")
        pylab.colorbar(pad=0., shrink=0.6)
        """Side view, Y-integrated"""
        pylab.subplot(2, 2, 1)
        plotMe = myScale(myProjection(scipy.transpose(
            myHist, (0,2,1))))[:,::-1]
        aspect = (
            ((xBins[-1] - xBins[0]) * 1.0 / (len(xBins) - 1)) * 1.0 /
            ((zBins[-1] - zBins[0]) * 1.0 / (len(zBins) - 1)))
        if nm_per_pixel != (None, None, None):
            aspect *= nm_per_pixel[0] * 1.0/nm_per_pixel[2]
        extent = (xyzSlice[2].stop, xyzSlice[2].start,
                  xyzSlice[0].stop, xyzSlice[0].start)
        pylab.imshow(
            plotMe, interpolation='nearest', cmap=pylab.cm.hot,
            vmin=vmin, vmax=vmax, aspect=aspect, extent=extent)
        pylab.title("Side view")
        pylab.colorbar(pad=0., shrink=0.6)
        pylab.gcf().show()
        pylab.ion()
        print (
            "\nAdjust (b)rightness, re-(s)lice, change (p)rojection, " +
            "change (f)ilter, toggle\n(l)og-linear, (o)utput raw data, " +
            "output s(m)oothed data, or (d)one? b/s/p/f/[l]/o/m/d:"),
        cmd = raw_input()
        if cmd == 'b':
            print "Minimum displayed brightness [image min]:",
            newmin = raw_input()
            try:
                vmin = float(newmin)
            except ValueError:
                vmin = None
            if vmin is not None:
                print "Maximum displayed brightness [image max]:",
                newmax = raw_input()
                try:
                    vmax = float(newmax)
                except ValueError:
                    vmax = None
            if vmin == None or vmax == None:
                print "Using image brightness range"
                (vmin, vmax) = (None, None)
            print "Min, max: %s, %s"%(vmin, vmax)
        elif cmd == 's':
            print "Top view, leftmost pixel:",
            leftmost = raw_input()
            print "Top view, rightmost pixel:",
            rightmost = raw_input()
            try:
                (leftmost, rightmost) = (int(leftmost), int(rightmost))
            except ValueError:
                print "Input not understood, using image limits."
                (leftmost, rightmost) = (0, histograms[whichFilter].shape[1])
            print "Top view, upper pixel:",
            highest = raw_input()
            print "Top view, lower pixel:",
            lowest = raw_input()
            try:
                (highest, lowest) = (int(highest), int(lowest))
            except ValueError:
                print "Input not understood, using image limits."
                (highest, lowest) = (0, histograms[whichFilter].shape[0])
            print "Front view, upper pixel:",
            loZ = raw_input()
            print "Front view, lower pixel:",
            hiZ = raw_input()
            try:
                (loZ, hiZ) = (int(loZ), int(hiZ))
            except ValueError:
                print "Input not understood, using image limits."
                (loZ, hiZ) = (0, histograms[whichFilter].shape[2])
            xyzSlice = (
                slice(highest, lowest),
                slice(leftmost, rightmost),
                slice(loZ, hiZ))
        elif cmd == 'p':
            print "(s)ummed projection or (m)aximum projection? [s]/m:",
            cmd = raw_input()
            if cmd == 'm':
                whichProjection = 'max'
            else:
                whichProjection = 'sum'
        elif cmd == 'f':
            while True:
                print "Filter number to view:",
                i = raw_input()
                try:
                    i = int(i)
                except ValueError:
                    print "Type an integer, hit return."
                    continue
                try:
                    hist = histograms[i]
                    whichFilter = i
                    break
                except IndexError:
                    print "Valid numbers are", range(len(histograms))
        elif cmd == 'o' or cmd == 'm':
            print "Data file name [histogram]:",
            dataFileName = raw_input()
            if dataFileName == '':
                dataFileName = 'histogram'
            if cmd == 'm':
                while True:
                    print "Smoothing sigma:",
                    sigma = raw_input()
                    try:
                        sigma = float(sigma)
                        break
                    except ValueError:
                        print "Type a number, hit return"
                from scipy.ndimage import gaussian_filter
                outputMe = gaussian_filter(myHist, sigma=sigma)
                dataType = scipy.float64
                typeDescription = "64-bit real"
            else:
                outputMe = myHist
                dataType = scipy.uint32
                typeDescription = "32-bit unsigned"
            scipy.transpose(outputMe, (2,0,1)).astype(dataType).tofile(
                open(dataFileName + '.dat', 'wb'))
            dataDetails = open(dataFileName + '.txt', 'w')
            dataDetails.write(
                "Image type: %s\n"%(typeDescription) +
                "Width: %i pixels (bins)\n"%(myHist.shape[1]) +
                "Height: %i pixels (bins)\n"%(myHist.shape[0]) +
                "Number of slices: %i\n"%(myHist.shape[2]) +
                "Little-Endian Byte Order\n" +
                "\n\n\n" +
                "Nanometers per camera pixel: " + repr(nm_per_pixel) + '\n' +
                "Up-down bin edges (camera pixel coordinates):\n" +
                repr(xBins) + '\n' +
                "Left-right bin edges (camera pixel coordinates):\n" +
                repr(yBins) + '\n' +
                "Axial bin edges: (calibration stack coordinates):\n" +
                repr(zBins) + '\n')
            dataDetails.close()
        elif cmd == 'd':
            print "Done displaying histogram."
            break
        else: ##cmd == 'l' assumed, so toggle logscale
            logScale = not logScale
            print "Toggling log scale to: %s"%(logScale)
        
    return histograms

def plot_slices(stack3d, labelList=None, showFigs=True):
    import pylab

    figList = []
    if labelList is None:
        labelList = ['%i'%(i) for i in range(stack3d.shape[2])]
    whichFigure = 0
    pylab.ioff()
    for whichSlice in range(stack3d.shape[2]):
        if whichSlice - 20*whichFigure >= 0:
            figList.append(pylab.figure())
            pylab.suptitle('Slices')
            whichFigure += 1

        pylab.subplot(4,5, 21 + whichSlice - 20*whichFigure)
        pylab.imshow(
            stack3d[:,:,whichSlice],
            interpolation='nearest'
            )
        pylab.xticks([])
        pylab.yticks([])
        pylab.title(labelList[whichSlice])
        whichSlice += 1

    pylab.ion()
    if showFigs:
        for fig in figList:
            fig.show()
    return figList

def dft_resample_n(
    x_ft, num, start_index=[0], samples = ['all'],
    t = [None], axes=[0], window=[None], real_output=True, ifft=False):
    """
    Apply dft_resample in more than one direction, one direction at a
    time.

    'x_ft' is the original data, fft'd along the resampling axes.
    'num', 'output_subslice', 't', and 'axes' are lists with the same
    number of entries, one for each dimension of 'x_ft' that is to be
    resampled. The output 'y' is a resampled version of 'x_ft', and
    'new_t' is a list of resampling coordinates, one for each
    resampled axis.
    """
    import scipy

    y = scipy.asarray(x_ft)
    new_t = [[]]*len(axes)
    for i, axis in enumerate(axes):
        if t[i] is None:
            old_t = range(x_ft.shape[axis])
        else:
            old_t = t[i]
        if ifft:
            (y, new_t[i]) = fft_resample(
                x=y, num=num[i],
                t = old_t, axis=axis, window=window[i])
        else:
            (y, new_t[i]) = dft_resample(
                x_ft=y, num=num[i],
                start_index=start_index[i], samples=samples[i],
                t = old_t, axis=axis, window=window[i], real_output=False)
    if real_output:
        y = y.real

    return (y, new_t)

def dft_resample(
    x_ft, num,
    start_index=0, samples = 'all',
    t=None, axis=0, window=None,
    real_output=True):
    """
    Similar to scipy.signal.resample, but using equivalent discrete
    Fourier transform (DFT) matrix multiplication instead of using the
    fast Fourier transform (FFT). Inputs are similar, except for
    'x_ft', which is the FFT of x along 'axis', and 'start_index',
    'samples', which give the index of the first sample and number of
    samples that will return. With start_index=0, samples='all' and
    x_ft=scipy.fft(x, axis=axis), this function should return
    identical results to scipy.signal.resample.

    Drawbacks: You lose all the speed advantages of the FFT over
    matrix-multiplication DFTs
    
    Advantages: The 'output_subslice' option. FFT-based resampling
    always computes the full resampled signal, which is enormous for
    large upsampling factors. If you're only interested in a subset of
    the resampled signal (for example, near the maximum), this
    algorithm doesn't spend time computing other values. This is
    especially important when upsampling large multi-dimensional
    arrays.
    """
    import scipy
    from scipy.fftpack import ifftshift
    from scipy.signal import get_window
    
    X = scipy.asarray(x_ft)
    Nx = X.shape[axis]
    if window is not None:
        W = ifftshift(get_window(window,Nx))
        newshape = scipy.ones(len(X.shape))
        newshape[axis] = len(W)
        W=W.reshape(newshape)
        X = X*W
    if samples == 'all':
        samples = num
    roi = slice(start_index%num, start_index%num + samples)
    N = int(min(Nx, num))
    sl_1 = slice(0, (N+1)//2)
    sl_2 = slice(-(N-1)//2, None)
    k1 = scipy.array(
        range(num)[sl_1] +
        [1j] * (Nx - num) +   ##for downsampling: see 'isreal' below
        range(num)[sl_2])
    n1 = scipy.array(range(num))
    k = scipy.tile(k1,(roi.stop,1))[roi, :]
    n = scipy.tile(scipy.transpose(scipy.tile(n1,(Nx,1)), (1,0)),
                   (1 + roi.stop//num, 1))[roi, :]
    ifft_mat = (
        ((1.0 / num) *
         scipy.exp(2.j*scipy.pi*k*n / num)) *
        scipy.isreal(k)) ##Zeros portions of ifft_mat for downsampling
    """scipy.dot() works on the 2nd-to-last dimension of the second
    array. Roll 'axis' there, multiply, roll back:"""
    X_t = scipy.rollaxis(
        scipy.rollaxis(X,axis,0),
        0, len(X.shape)-1)
    y_t = scipy.dot(ifft_mat, X_t) * (num * 1.0 / Nx)
    y = scipy.rollaxis(y_t, 0, axis+1)
    if real_output:
        y = y.real
    if t is None:
        return y
    else:
        new_t = scipy.arange(roi.start, roi.stop) * (t[1]-t[0]) * 1.0 * Nx / num + t[0]
        return y, new_t

def fft_resample(x,num,t=None,axis=0,window=None):
    """Lifted nearly verbatim from scipy.signal.resample. The only
    difference is we skip the initial fft
    """
    from scipy import asarray, minimum, zeros, ifft, arange
    from scipy.fftpack import ifftshift
    from scipy.signal import get_window

    X = asarray(x)
    X = x
    Nx = x.shape[axis]
    if window is not None:
        W = ifftshift(get_window(window,Nx))
        newshape = ones(len(x.shape))
        newshape[axis] = len(W)
        W=W.reshape(newshape)
        X = X*W
    sl = [slice(None)]*len(x.shape)
    newshape = list(x.shape)
    newshape[axis] = num
    N = int(minimum(num,Nx))
    Y = zeros(newshape,'D')
    sl[axis] = slice(0,(N+1)/2)
    Y[sl] = X[sl]
    sl[axis] = slice(-(N-1)/2,None)
    Y[sl] = X[sl]
    y = ifft(Y,axis=axis)*(float(num)/float(Nx))

    if x.dtype.char not in ['F','D']:
        y = y.real

    if t is None:
        return y
    else:
        new_t = arange(0,num)*(t[1]-t[0])* Nx / float(num) + t[0]
        return y, new_t

def normalize_slices(data):
    """Given a stack of 2D images, normalizes each image to have unit
    vector magnitude. Complex data is allowed.

    The last coordinate in data_3d stores the image index:
    image_0 = data_3d[:,:,0], for example.
    """
    import scipy

    data_3d = scipy.atleast_3d(data)
    norm_3d = 1j * scipy.zeros(data_3d.shape) ##To handle complex data
    for whichSlice in range(data_3d.shape[2]):
        mag = scipy.sqrt((abs(data_3d[:,:,whichSlice])**2).sum())
        norm_3d[:,:,whichSlice] = data_3d[:,:,whichSlice] * 1./ mag
    return norm_3d.squeeze()

def human_sorted(myList):
    import re
    """ Sort the given list in the way that humans expect. 
    """ 
    def convert(text):
      return int(text) if text.isdigit() else text 
    def alphanum_key(key):
      return [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(myList, key=alphanum_key)

##def plot_cutplanes(data_3d):
##    from enthought.mayavi import mlab
##    import scipy
##
##    (xMax, yMax, zMax) = scipy.unravel_index(
##        data_3d.argmax(),
##        data_3d.shape
##        )
##    print (xMax, yMax, zMax)
##    source = mlab.pipeline.scalar_field(data_3d)
##    mlab.pipeline.image_plane_widget(
##        source,
##        plane_orientation='x_axes',
##        slice_index=xMax,
##        )
##    mlab.pipeline.image_plane_widget(
##        source,
##        plane_orientation='y_axes',
##        slice_index=yMax,
##        )
##    mlab.pipeline.image_plane_widget(
##        source,
##        plane_orientation='z_axes',
##        slice_index=zMax,
##        )
##    mlab.title(
##        '%i, %i, %i'%(xMax, yMax, zMax),
##        size=0.4,
##        height=0.9
##        )
##    mlab.colorbar()
##    mlab.outline()
##
##    return None

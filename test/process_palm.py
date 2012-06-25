import Tkinter, tkFileDialog, os, sys, time
import pylab
import palm_3d

tkroot = Tkinter.Tk()
tkroot.withdraw()
print "Where was the data spooled?"
dataDir = os.path.abspath(
    tkFileDialog.askdirectory(title="Where was the data spooled?"))
tkroot.destroy()
os.chdir(dataDir)

orphanDir = os.path.join(dataDir, 'orphans')
if not os.path.isdir(orphanDir):
    print "Making directory for orphan data"
    os.mkdir(os.path.join(dataDir, orphanDir))

##while not os.path.isfile(os.path.join(dataDir, 'palm_3d.py')):
##    print "Please put a copy of palm_3d.py in the data directory",
##    print "then hit enter."
##    raw_input()
##import palm_3d

def get_andor_message():
    andor_message_filename = (
        "C:/Documents and Settings/User/My Documents" +
        "/andor_scripts/andor_messages.txt")
    if os.path.isfile(andor_message_filename):
        time.sleep(1)
        andor_message_file = open(andor_message_filename, 'rb')
        andor_message = andor_message_file.read()
        andor_message_file.close()
        os.remove(andor_message_filename)
        return andor_message.splitlines()[-1].strip()
    else:
        return None

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
    (yPix, xPix) = (1 + int(right) - int(left), 1 + int(top) - int(bottom))
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
        f, count=xPix*yPix, dtype = numpy.float32).reshape(xPix, yPix)
    f.close()
    return image

def sif_to_dat(sifName, datName, saveTxt=True, progressBar=False):
    import numpy

    dtypeOut = numpy.uint16
    datFile = open(datName, 'wb')
    xPix, yPix, numImages, offset = get_sif_info(sifName)
    if progressBar and numImages > 300:
        import sys
    for i in range(numImages):
        im = numpy.atleast_3d(
            load_sif_image(sifName, i, xPix, yPix, offset))
        im.transpose(2, 0, 1).astype(dtypeOut).tofile(datFile)
        if i%20 == 19:
            if progressBar and numImages > 300:
                sys.stdout.write(
                    "\r Converting image: %06i/%i"%(i + 1, numImages))
                sys.stdout.flush()
    if progressBar and numImages > 300:
        sys.stdout.write(
            "\r Converting image: %06i/%i\n"%(i + 1, numImages))
        sys.stdout.flush()
    datFile.close()
    if saveTxt:
        import os
        datFileMinusExtension = os.path.splitext(datName)[0]
        dataDetails = open(datFileMinusExtension + '.txt', 'w')
        dataDetails.write(
            "Image type: %s\n"%(repr(dtypeOut)) +
            "Width: %i pixels\n"%(yPix) +
            "Height: %i pixels\n"%(xPix) +
            "Number of slices: %i\n"%(numImages) +
            "Little-Endian Byte Order\n")
        dataDetails.close()
    return (xPix, yPix, numImages, offset)

def convert_all_sifs(dirName='os.getcwd()', saveTxt=True):
    """Recursively converts all Andor .sif files to unsigned 16-bit
    raw binary files. If saveTxt==True, will also generate text
    descriptions of the raw binary files, useful for loading into
    ImageJ."""
    import os

    if dirName == 'os.getcwd()':
        dirName = os.getcwd()
    dirName = os.path.abspath(dirName)
    directoryContents = sorted(os.listdir(dirName))
    for f in directoryContents:
        f = os.path.join(dirName, f)
        baseName, ext = os.path.splitext(f)
        if ext == '.sif':
            print "Converting [", f, "], creating:"
            print "*", baseName + '.dat'
            if saveTxt:
                print "*", baseName + '.txt'
            sif_to_dat(f, baseName + '.dat', saveTxt=saveTxt, progressBar=True)
        elif os.path.isdir(f):
            print "\nDescending into directory", f
            convert_all_sifs(dirName=os.path.join(dirName, f), saveTxt=saveTxt)
        else:
            print "Not converting", f
    return None

print "Waiting for Andor message..."
while True:
    time.sleep(3)
    andor_message = get_andor_message()
    if andor_message is None:
        continue
    print "Andor message:", andor_message
    if andor_message == 'calibration':
        calibrationDir = os.path.join(dataDir, 'calibration')
        if os.path.isdir(calibrationDir):
            print "Calibration directory already exists"
        else:
            os.mkdir(calibrationDir)
        sifsInDataDir = sorted([i for i in os.listdir(dataDir)
                         if os.path.splitext(i)[-1] == '.sif'])
        calImages = []
        for sif in sifsInDataDir:
            (xPix, yPix, numImages, offset) = get_sif_info(
                os.path.join(dataDir, sif))
            if numImages == 10:
                os.rename(os.path.join(dataDir, sif),
                          os.path.join(calibrationDir, sif))
                for i in range(10):
                    calImages.append(os.path.splitext(sif)[0] + '.dat*%i'%(i))
            else:
                print "Moving orphan data:", sif
                os.rename(os.path.join(dataDir, sif),
                          os.path.join(orphanDir, sif))
        convert_all_sifs(dirName=calibrationDir)
        cal_xy_shape = (xPix, yPix)
        for cal in calImages:
            print cal
        print cal_xy_shape
        data = palm_3d.new_palm(
            images=[],
            imFolder='./',
            calImages=calImages,
            calFolder='./calibration',
            cal_format='raw',
            cal_xy_shape=cal_xy_shape,
            filename_prefix = 'DELETEME_')
        data.load_calibration(
            calibrationRepetitions=2,
            calibrationImagesPerPosition=10,
            smoothing_sigma=(1,1,5),
            promptForSave=False)
        possiblePoop = os.path.join(dataDir, 'DELETEME_palm_acquisition.pkl')
        if os.path.isfile(possiblePoop):
            os.remove(possiblePoop)
    elif andor_message.split()[0] == 'slice_position:':
        print "Processing PALM data..."
        slicePosition = float(andor_message.split()[1])
        repetitions = int(andor_message.split()[3])
        sliceshots = int(andor_message.split()[5])
        fidshots = int(andor_message.split()[7])
        formattedSlicePosition = ('%3.2f'%(slicePosition*0.1)
                                  ).replace('.', 'p').replace('-', 'n')
        newDirNum = 1
        while True:
            newDir = 'z=' + formattedSlicePosition + 'um_' + repr(newDirNum)
            if os.path.isdir(os.path.join(dataDir, newDir)):
                newDirNum += 1
            else:
                print "Creating directory ", newDir
                os.mkdir(os.path.join(dataDir, newDir))
                break
        sifsInDataDir = sorted([i for i in os.listdir(dataDir)
                 if os.path.splitext(i)[-1] == '.sif'])
        images = []
        im_z_positions = {}
        for sif in sifsInDataDir:
            (xPix, yPix, numImages, offset) = get_sif_info(
                os.path.join(dataDir, sif))
            if numImages == sliceshots or numImages == fidshots:
                os.rename(os.path.join(dataDir, sif),
                          os.path.join(newDir, sif))
                for i in range(numImages):
                    images.append(os.path.splitext(sif)[0] + '.dat*%i'%(i))
                    if numImages == sliceshots:
                        im_z_positions[images[-1]] = slicePosition * 2
                    elif numImages == fidshots:
                        im_z_positions[images[-1]] = 0
            else:
                print "Moving orphan data:", sif
                os.rename(os.path.join(dataDir, sif),
                          os.path.join(orphanDir, sif))
        convert_all_sifs(dirName=newDir)
        filename_prefix = formattedSlicePosition + 'um_' + repr(newDirNum) + '_'
        print "Writing metadata..."
        metaDataFile = open(
            os.path.join(newDir, 'metadata_oldstyle.txt'), 'wb')
        metaDataFile.write(andor_message)
        metaDataFile.close()
##We skip writing the localization script, since palm_gui does that now.
##        print "Writing localization script..."
##        locScript = open(
##            os.path.join(dataDir, filename_prefix + 'localization.py'), 'wb')
##        locScript.write(
##            "import palm_3d\n\n" +
##            "data = palm_3d.new_palm(\n" +
##            "    images=%s,\n"%repr(images) +
##            "    imFolder='./" + newDir + "',\n" +
##            "    im_format='raw',\n" +
##            "    im_xy_shape=(%s, %s),\n"%(repr(xPix), repr(yPix)) +
##            "    im_z_positions=%s,\n"%(repr(im_z_positions)) +
##            "    calImages=[],\n" +
##            "    calFolder='./calibration',\n" +
##            "    cal_format='raw',\n" +
##            "    cal_xy_shape=(%s, %s),\n"%(repr(xPix), repr(yPix)) +
##            "    filename_prefix = '%s')\n\n"%(filename_prefix) +
##            "data.load_calibration(\n" +
##            "    calibrationRepetitions=2,\n" +
##            "    calibrationImagesPerPosition=10,\n" +
##            "    smoothing_sigma=(1,1,5),\n" +
##            "    promptForSave=True)\n\n" +
##            "data.images_to_candidates()\n" +
##            "data.localize_candidates(promptForInspection=False)\n" +
##            "data.link_localizations(promptForInspection=False)\n" +
##            "data.localize_candidates(linkedInput=True, promptForInspection=False)\n" +
##            "data.select_fiducials()\n"
##            )
##        locScript.close()
        
    print "Waiting for Andor message..."


print "Hit enter to continue..."
raw_input()


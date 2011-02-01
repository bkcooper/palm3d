"""Graphical user interface for palm_3d. Version 0.1.2, build 7
Written by Andrew York and Kenneth Arcieri"""
import os, re, subprocess, palm_3d
import tkFileDialog, tkSimpleDialog, tkMessageBox, tkFont
import Tkinter as tk

basedir = os.getcwd()
def ipython_run(script, waitForIt=False):
    if os.name == 'posix':
        cmd = ['gnome-terminal', '-t', 'palm3d subprocess', '--disable-factory',
             '-e', """ipython -pylab -c 'run "%s"'"""%(script)]
    elif os.name == 'nt':
        cmd = [
            os.path.join(basedir, "portablepython", "App", "python.exe"),
            os.path.join(
                basedir, "portablepython", "App", "Scripts", "ipython"),
            "-pylab", "-c", 'run "%s"'%(script)]
    print cmd
    proc = subprocess.Popen(cmd)
    if waitForIt:
        proc.communicate()
    return proc

class Gui:
    def __init__(self, root):
        root.report_callback_exception = self.report_callback_exception
        root.title("3D palm data processing")
        root.minsize(680, 0)
        
        """
        Make the root frame scrollable, and populate it with widgets
        """
        self.unscrollable_root = root ##Important for tkSimpleDialog
        self.add_scrollable_root(root)
        self.root.update_idletasks()
        """
        Create the experiment folder selection toolbar
        """
        b = tk.Label(self.root, text="Experiment folder:")
        b.grid(columnspan=2)

        self.experimentFolderLabel = tk.Label(
            self.root,
            text=" ",
            justify=tk.LEFT)
        self.experimentFolderLabel.grid(columnspan=2)

        b = tk.Button(
            self.root, text="Select", command=self.select_experiment_folder)
        b.focus_set()
        b.bind("<Return>", self.select_experiment_folder)
        b.grid(row=0, column=2, rowspan=2)
        """
        Create the calibration and image folder selection toolbars
        """
        b = tk.Label(self.root, text=" Calibration folder name:")
        b.grid()
        self.calFolderEntry = tk.StringVar()
        b = tk.Entry(self.root, textvariable=self.calFolderEntry)
        b.bind("<Return>", self.refresh_data_display)
        self.calFolderEntry.set("calibration")
        b.grid(row=2, column=1)

        b = tk.Label(self.root, text=" Image folder prefix:")
        b.grid()
        self.dataPrefixEntry = tk.StringVar()
        b = tk.Entry(self.root, textvariable=self.dataPrefixEntry)
        b.bind("<Return>", self.refresh_data_display)
        self.dataPrefixEntry.set("z=")
        b.grid(row=3, column=1)

        b = tk.Button(
            self.root, text="Refresh", command=self.refresh_data_display)
        b.bind("<Return>", self.refresh_data_display)
        b.grid(row=2, column=2, rowspan=2)
        """
        Create the data display
        """
        self.dataDisplay = tk.Frame(self.root, bd=1, relief=tk.SUNKEN)
        self.dataDisplay.grid(columnspan=3, sticky='nsew')
        self.root.grid_rowconfigure(3, weight=1)
        self.root.grid_columnconfigure(0, weight=1)
        self.select_experiment_folder()
        """Configure size of canvas's scrollable zone"""
        self.resize_scrollregion()
        return None

    def report_callback_exception(self, *args):
        import traceback
        err = traceback.format_exception(*args)
        with open(os.path.join(
            self.experimentFolder,
            'error_log.txt'), 'ab') as error_log:
            for e in err:
                error_log.write(e + os.linesep)
            error_log.write(os.linesep*2)
        tkMessageBox.showerror(
            'Exception',
            'An exception occured. ' +
            'Read "error_log.txt" in the experiment folder for details."')
        return None

    def add_scrollable_root(self, root):
        """
        A little hack to make the root window scrollable.

        Adds a canvas 'self.cnv' to the root window, with scrollbars,
        and a frame 'self.root' to this canvas. After modifying the
        contents of 'self.root', call 'self.resize_scrollregion()' to
        make sure the scrolling region covers the contents of
        'self.root'.

        """
        """Grid sizing behavior in window"""
        root.grid_rowconfigure(0, weight=1)
        root.grid_columnconfigure(0, weight=1)
        """Canvas with scrollbars"""
        self.cnv = tk.Canvas(root)
        self.cnv.grid(row=0, column=0, sticky='nswe')
        hScroll = tk.Scrollbar(
            root, orient=tk.HORIZONTAL, command=self.cnv.xview)
        hScroll.grid(row=1, column=0, sticky='we')
        vScroll = tk.Scrollbar(
            root, orient=tk.VERTICAL, command=self.cnv.yview)
        vScroll.grid(row=0, column=1, sticky='ns')
        self.cnv.configure(
            xscrollcommand=hScroll.set, yscrollcommand=vScroll.set)
        """Frame in canvas is now scrollable - pretend it's the root window"""
        self.root = tk.Frame(self.cnv)
        self.cnv.create_window(0, 0, window=self.root, anchor='nw')
        return None

    def resize_scrollregion(self):
        self.root.update_idletasks()
        self.cnv.configure(scrollregion=(
            0, 0, self.root.winfo_width(), self.root.winfo_height()))
        return None

    def select_experiment_folder(self, event=None):
        self.experimentFolder = tkFileDialog.askdirectory(
            title="Where is the PALM data?")
        if self.experimentFolder == '' or self.experimentFolder == ():
            self.experimentFolder = os.getcwd()
        print self.experimentFolder
        os.chdir(self.experimentFolder)
        self.experimentFolderLabel['text'] = self.experimentFolder + " " * 3
        self.refresh_data_display()
        return None

    def refresh_data_display(self, event=None):
        print "Refreshing data display"
        """
        First, remove old displayed data
        """
        for s in self.dataDisplay.grid_slaves():
            s.destroy()
        """
        Show status of calibration data
        """
        calFolder = os.path.abspath(os.path.join(
            self.experimentFolder, self.calFolderEntry.get()))
        b = tk.Label(
            self.dataDisplay,
            text=(
                ("No calibration data found in experiment directory.\n" +
                 "Change the calibration folder name" +
                 " or select another directory."),
                "Calibration folder:\n./" + os.path.relpath(calFolder)
                )[os.path.exists(calFolder)],
            justify=tk.CENTER)
        b.grid(columnspan=(4, 1)[os.path.exists(calFolder)])
        self.calLoaded = os.path.exists(
            os.path.join(calFolder, 'calibration.pkl'))
        if os.path.exists(calFolder):
            b = tk.Button(
                self.dataDisplay,
                text=("Load", "Loaded")[self.calLoaded],
                command=(self.load_calibration, None)[self.calLoaded],
                state=(tk.NORMAL, tk.DISABLED)[self.calLoaded])
            b.bind("<Return>", (self.load_calibration, None)[self.calLoaded])
            b.grid(row=0, column=1)
        """
        Show status of image data
        """
        self.get_acquisition_data()
        if len(self.dataFolders) == 0:
            b = tk.Label(
                self.dataDisplay,
                text=("\nNo image data found in current experiment directory." +
                      "\nChange the image folder prefix" +
                      " or select another directory.\n"),
                justify=tk.LEFT)
            b.grid(row=1, column=0)
        else:
            for i, label in enumerate((
                "Acquisition  ",
                "File prefix  ",
                'Z pos.   ',
                'Dimensions   ',
                'Processing   ',
                'Progress   ')):
                b = tk.Label(
                    self.dataDisplay,
                    text=label,
                    font=tkFont.Font(size=9, weight=tkFont.BOLD),
                    justify=tk.LEFT)
                b.grid(row=1, column=i, sticky='w')
            metadataDefaults = {}
            anyFinished = False
            for f, fol in enumerate(self.dataFolders):
                b = tk.Label(
                    self.dataDisplay,
                    text=fol["folder"],
                    justify=tk.LEFT)
                b.grid(row=f+2, column=0, sticky='w')

                b = tk.Label(
                    self.dataDisplay,
                    text=fol["file_prefix"],
                    justify=tk.LEFT)
                b.grid(row=f+2, column=1, sticky='w')
                
                b = tk.Label(
                    self.dataDisplay,
                    text=fol.get('slice_z_position:','?'))
                b.grid(row=f+2, column=2, sticky='w')

                b = tk.Label(
                    self.dataDisplay,
                    text=(
                        repr(fol.get('x_pixels:', 0)) + 'x' +
                        repr(fol.get('y_pixels:', 0)) + 'x' +
                        repr(int(
                        fol.get('repetitions:', 0) * (
                            fol.get('slice_images_per_file:', 0) +
                            fol.get('tracking_images_per_file:', 0))))
                        ))
                b.grid(row=f+2, column=3, sticky='w')

                hasMetadata = (
                    'slice_z_position:' and 'repetitions:' and
                    'slice_images_per_file:' and 'tracking_images_per_file' and
                    'x_pixels:' and 'y_pixels:' in fol)
                if hasMetadata:
                    metadataDefaults = fol
                b = tk.Button(
                    self.dataDisplay,
                    text=("Set metadata", "Process")[hasMetadata],
                    command=(
                        lambda f=fol, m=metadataDefaults: self.set_metadata(
                            folder=f['folder'],
                            keys=['slice_z_position:', 'repetitions:',
                                  'slice_images_per_file:',
                                  'tracking_images_per_file:',
                                  'x_pixels:', 'y_pixels:'],
                            defaults=m),
                        lambda f=fol: self.process_palm(f))[hasMetadata],
                    state=(tk.DISABLED, tk.NORMAL)[self.calLoaded])
                b.bind("<Return>", (
                    None, lambda e, f=fol: self.process_palm(f))[hasMetadata])
                b.grid(row=f+2, column=4, sticky='w')

                b = tk.Label(
                    self.dataDisplay,
                    text=(fol['progress']))
                b.grid(row=f+2, column=5, sticky='w')

                if fol['progress'] == 'Done':
                    anyFinished = True
                    fol['include_in_histogram'] = tk.IntVar()
                    fol['include_in_histogram'].set(1)
                    b = tk.Checkbutton(
                        self.dataDisplay,
                        variable=fol['include_in_histogram'])
                    b.grid(row=f+2, column=6, sticky='w')
                    
##                    b = tk.Button(
##                        self.dataDisplay,
##                        text='?',
##                        justify=tk.LEFT,
##                        command=(
##                            lambda f=fol: self.inspection(f)))
##                    b.grid(row=f+2, column=7, sticky='w')
            if anyFinished:
                b = tk.Button(
                    self.dataDisplay,
                    text='Construct\nhistogram',
                    font=tkFont.Font(size=6, weight=tkFont.BOLD),
                    justify=tk.LEFT,
                    command=self.construct_histogram)
                b.bind("<Return>", self.construct_histogram)
                b.grid(row=1, column=6, sticky='w')

                b = tk.Button(
                    self.dataDisplay,
                    text='Invert\nselection',
                    font=tkFont.Font(size=6, weight=tkFont.BOLD),
                    justify=tk.LEFT,
                    command=self.invert_histogram_selection)
                b.bind("<Return>", self.invert_histogram_selection)
                b.grid(row=2+len(self.dataFolders), column=6, sticky='w')

        self.resize_scrollregion()
        print "Data display refreshed."
        return None

    def get_acquisition_data(self):
        """palm_3d.load_palm() is too expensive to call every time we
        refresh the palm_gui window. Store the results for faster
        loading."""
        old_load_cache = dict(getattr(self, 'load_palm_cache', {}))
        self.load_palm_cache = {}
        """Keys: Palm acquisition filenames. Values: (acq, mtime),
        where acq is the loaded palm acquisition, and mtime is the
        time that file was last modified."""
        """
        Get acquisition name and file prefix:
        """
        self.dataFolders = [
            {"folder": folder} for folder in
            human_sorted(os.listdir(self.experimentFolder))
            if (os.path.isdir(folder) and
                folder.startswith(self.dataPrefixEntry.get()))]
        goodFilenames = True
        for fol in self.dataFolders:
            fol["file_prefix"] = (
                fol["folder"].lstrip(self.dataPrefixEntry.get()) + '_')
            if not fol["file_prefix"].replace('_', '').isalnum():
                goodFilenames = False
            """
            Get metadata:
            """
            keys = (
                'slice_z_position:',
                'repetitions:',
                'slice_images_per_file:',
                'tracking_images_per_file:',
                'x_pixels:',
                'y_pixels:')
            metadata = self.get_metadata(folder=fol["folder"], keys=keys)
            if metadata is not None:
                for k, v in metadata.items():
                    fol[k] = v
            """
            Get progress information:
            """
            acq = fol['file_prefix'] + 'palm_acquisition.pkl'
            try:
                mtime = os.path.getmtime(acq)
            except OSError:
                fol['progress'] = ''
                continue
            if (acq in old_load_cache and mtime == old_load_cache[acq][1]):
                data = old_load_cache[acq][0]
                print "Using stored palm data"
            else:
                try:
                    data = palm_3d.load_palm(acq, verbose=False)
                    print "Loading palm data"
                except IOError:
                    fol['progress'] = ''
                    continue
            self.load_palm_cache[acq] = (data, mtime)
            fol['progress'] = 'Started'
            for attr, prog in (
                ('candidates_filename', 'Candidate selection'),
                ('localizations_filename', 'Localizing candidates'),
                ('linked_localizations_filename', 'Linking'),
                ('particles_filename', 'Relocalizing')):
                if os.path.exists(os.path.join(
                    data.imFolder, getattr(data, attr, 'xNOSUCHFILE'))):
                    fol['progress'] = prog
            if os.path.exists(getattr(
                data, 'fiducial_filter_filename', 'NOSUCHFILE')):
                fol['progress'] = 'Drift correction'
            if getattr(data, 'drift', None) != None:
                fol['progress'] = 'Done'
        if not goodFilenames:
            tkMessageBox.showerror(
                'Warning',
                'At least one file prefix has characters that are neither' +
                ' alphanumeric nor underscores. Change the offending' +
                ' acquisition folder name.')
        return None

    def get_metadata(self, folder, keys, dtype=int):
        """
        Load the metadata:
        """
        try:
            metadataFile = open(os.path.join(folder, 'metadata.txt'), 'rb')
            metadata = metadataFile.read().split()
            metadataFile.close()
        except IOError:
            metadata = []
        metadata = dict(zip(metadata[::2], metadata[1::2]))
        try:
            for k in keys:
                metadata[k] = dtype(metadata[k])
        except (KeyError, ValueError):
            return None
        return metadata

    def set_metadata(self, folder, keys, defaults=None):
        if defaults is None:
            defaults = {}
        responses = {}
        for k in keys:
            responses[k] = tkSimpleDialog.askstring(
                title=os.path.split(folder)[-1], prompt=k,
                initialvalue=str(defaults.get(k, '')),
                parent=self.unscrollable_root)
            if responses[k] is None:
                return responses
        with open(os.path.join(folder, 'metadata.txt'), 'wb') as mdFile:
            for k in keys:
                mdFile.write(k + " " + responses.get(k, '') + os.linesep)
        self.refresh_data_display()
        return responses

    def load_calibration(self, event=None):
        calFolder = os.path.abspath(os.path.join(
            self.experimentFolder, self.calFolderEntry.get()))
        keys = ('calibration_repetitions:',
                'calibration_images_per_position:',
                'x_pixels:',
                'y_pixels:')
        defaults = {'calibration_repetitions:': 2,
                    'calibration_images_per_position:': 10}
        metadata = self.get_metadata(folder=calFolder, keys=keys)
        if metadata is None:
            self.set_metadata(folder=calFolder, keys=keys, defaults=defaults)
            metadata = self.get_metadata(folder=calFolder, keys=keys)
        if metadata is None: ##The user probably canceled
            return None
        """
        Load the calibration data:
        """
        with open(os.path.join(calFolder, 'calibration.py'), 'wb') as calScript:
            calScript.write(
"""try:
    import os, palm_3d

    calFolder = {calFolder}
    calImages = []
    print "\\nCalibration files:"
    for f in palm_3d.human_sorted(os.listdir(calFolder)):
        if os.path.splitext(f)[-1] == '.dat':
            print f
            for i in range({imPerPos}):
                calImages.append(f+'*%i'%(i))    

    data = palm_3d.new_palm(
        images=[],
        imFolder={experimentFolder},
        calImages=calImages,
        calFolder=calFolder,
        cal_format='raw',
        cal_xy_shape={cal_xy_shape},
        filename_prefix = 'DELETEME_')
    data.load_calibration(
        calibrationRepetitions={calReps},
        calibrationImagesPerPosition={imPerPos},
        smoothing_sigma=(1,1,5),
        promptForSave=False)
    import pylab
    pylab.close('all') ##To prevent a funky Windows error
except:
    import traceback
    traceback.print_exc()
    raw_input()
                """.format(
                    calFolder=repr(calFolder),
                    experimentFolder=repr(self.experimentFolder),
                    cal_xy_shape=(metadata['x_pixels:'], metadata['y_pixels:']),
                    calReps=metadata['calibration_repetitions:'],
                    imPerPos=metadata['calibration_images_per_position:']
                    ).replace('\n', os.linesep))
        ipython_run(os.path.join(calFolder, 'calibration.py'), waitForIt=True)
        possiblePoop = os.path.join(
            self.experimentFolder, 'DELETEME_palm_acquisition.pkl')
        if os.path.isfile(possiblePoop):
            os.remove(possiblePoop)
        self.refresh_data_display()
        return None

    def process_palm(self, acquisition):
        acq = acquisition
        print acq
        """
        Write the localization script:
        """
        scriptName = acq['file_prefix']+'localization.py'
        with open(scriptName, 'wb') as locScript:
            locScript.write(
"""try:
    import os, palm_3d

    im_folder = {imFolder}
    print "Image acquisition folder:", im_folder, '\\n'
    slice_z_position = {slicePosition}
    repetitions = {repetitions}
    slice_images_per_file = {imPerPos}
    tracking_images_per_file = {trImPerPos}
    x_pixels, y_pixels = {xyShape}
    images = [] #The list of data files
    im_z_positions = {{}} #The z-position at which each data file was taken
    num = 0
    if tracking_images_per_file > 0:
        num_expected = 2 * repetitions
    else:
        num_expected = repetitions
    print "\\nImage files:"
    for f in palm_3d.human_sorted(os.listdir(im_folder)):
        if os.path.splitext(f)[-1] == '.dat':
            print f
            num += 1
            if num > num_expected:
                print "\\nExtra images found!"
                print "Check that 'repetitions' is correct.\\n"
                break
            if num%2 == 1: #Every other file is slice data
                for i in range(slice_images_per_file):
                    images.append(f+'*%i'%(i))
                    im_z_positions[images[-1]] = slice_z_position
            else: #Every OTHER other file is tracking data
                for i in range(tracking_images_per_file):
                    images.append(f+'*%i'%(i))
                    im_z_positions[images[-1]] = 0
    if num < num_expected:
        print "\\nFewer images than expected!"
        print "Check that 'repetitions' is correct.\\n"

    data = palm_3d.new_palm(
        images=images,
        imFolder=im_folder,
        im_format='raw',
        im_xy_shape=(x_pixels, y_pixels),
        im_z_positions=im_z_positions,
        calFolder={calFolder},
        filename_prefix = {filePrefix})
    data.load_calibration(promptForSave=False, promptForInspection=False)
    data.save(overwriteWarning=False)
    data.images_to_candidates()
    data.localize_candidates(promptForInspection=False)
    data.link_localizations(promptForInspection=False)
    data.localize_candidates(linkedInput=True, promptForInspection=False)
    data.select_fiducials()
    import pylab
    pylab.close('all') ##To prevent a funky Windows error message
    raw_input("Hit enter to exit...")
except:
    import traceback
    traceback.print_exc()
    raw_input()
                """.format(
                    imFolder=repr(os.path.abspath(acq['folder'])),
                    xyShape=(acq['x_pixels:'], acq['y_pixels:']),
                    reps=acq['repetitions:'],
                    imPerPos=acq['slice_images_per_file:'],
                    trImPerPos=acq['tracking_images_per_file:'],
                    slicePosition=acq['slice_z_position:'],
                    repetitions=acq['repetitions:'],
                    calFolder=repr(os.path.abspath(os.path.join(
                        self.experimentFolder, self.calFolderEntry.get()))),
                    filePrefix=repr(acq['file_prefix'])
                    ).replace('\n', os.linesep))
        ipython_run(scriptName)
        self.refresh_data_display()
        return None

    def invert_histogram_selection(self, event=None):
        for f in self.dataFolders:
            if f.get('include_in_histogram', False): #Variable exists
                if f.get('include_in_histogram').get(): #Variable is true
                    f.get('include_in_histogram').set(False)
                else:
                    f.get('include_in_histogram').set(True)
        return None

    def construct_histogram(self, event=None):
        acquisitions = []
        for f in self.dataFolders:
            if f.get('include_in_histogram', False): #Variable exists
                if f.get('include_in_histogram').get(): #Variable is true
                    acquisitions.append(f)
        if len(acquisitions) == 0:
            return None
        acquisitionStr = ''.join(
            [' '*8 + "'" + f['file_prefix'] + "palm_acquisition.pkl',\n"
             for f in acquisitions])
        data = palm_3d.load_palm(
            acquisitions[0]['file_prefix'] + 'palm_acquisition.pkl',
            verbose=False)
        xyShape = data.im_xy_shape
        numCalSlices = data.calibration.shape[2]
        keys = ('minimum_correlation:',
                'minimum_z_calibration:',
                'maximum_z_calibration:',
                'nanometers_per_x_pixel:',
                'nanometers_per_y_pixel:',
                'nanometers_per_z_pixel:',
                'nanometers_per_histogram_bin:',
                'minimum_x:',
                'maximum_x:',
                'minimum_y:',
                'maximum_y:',
                'minimum_z:',
                'maximum_z:')
        metadata = self.get_metadata(
            folder=self.experimentFolder, keys=keys, dtype=float)
        if metadata is not None:
            useOldMetadata = tkMessageBox.askyesnocancel(
                title="Histogram", message="Use old parameters?")
            if useOldMetadata is None:
                return None
            if not useOldMetadata:
                metadata = None
        if metadata is None:
            self.set_metadata(
                folder=self.experimentFolder, keys=keys,
                defaults={'minimum_correlation:': 0.4,
                          'minimum_z_calibration:': 10,
                          'maximum_z_calibration:': numCalSlices-10,
                          'nanometers_per_histogram_bin:': 100,
                          'minimum_x:': 0,
                          'maximum_x:': xyShape[0],
                          'minimum_y:': 0,
                          'maximum_y:': xyShape[1],
                          'minimum_z:': 0,
                          'maximum_z:': numCalSlices
                          })
            metadata = self.get_metadata(
                folder=self.experimentFolder, keys=keys, dtype=float)
            if metadata is None:
                return None
        linkedInput = tkMessageBox.askyesnocancel(
            title="Linking", message="Use linked input?")
        if linkedInput is None:
            return None
        with open('plots.py', 'wb') as histScript:
            histScript.write(
"""try:
    import palm_3d, scipy

    palmAcquisitions = [
{acquisitionStr}    ] ##Set this

    def myFilt_0(loc): ##Change this if required
        if loc['qual'] < {minQual}:
            return False
        if loc['z'] < {minCalZ}:
            return False
        if loc['z'] > {maxCalZ}:
            return False
        return True
    locFilters = [[myFilt_0]] * len(palmAcquisitions) ##Change this if required
    nm_per_pixel = {nm_per_pixel} ##up-down, left-right, axial
    nm_per_bin = {nm_per_bin}
    pixels_per_bin = nm_per_bin * 1.0 / scipy.array(nm_per_pixel)

    histograms = palm_3d.combine_palm_histograms(
        palmAcquisitions,
        locFilters=locFilters,
        xBins=scipy.arange({minX}, {maxX}, pixels_per_bin[0]),
        yBins=scipy.arange({minY}, {maxY}, pixels_per_bin[1]),
        zBins=scipy.arange({minZ}, {maxZ}, pixels_per_bin[2]),
        nm_per_pixel=nm_per_pixel,
        linkedInput={linkedInput},
        persistent=False) ##Set this

    import pylab
    pylab.close('all') ##To prevent a funky Windows error message
except (MemoryError, ValueError):
    import traceback
    traceback.print_exc()
    print "\\nA memory error or value error",
    print "often means the histogram is too big."
    print "You can make a smaller histogram by:"
    print "(a) Increasing 'nanometers_per_histogram_bin'"
    print "(b) Increasing 'minimum_x', 'minimum_y', or 'minimum_z'"
    print "(c) Decreasing 'maximum_x', 'maximum_y', or 'maximum_z'"
    raw_input()
except:
    import traceback
    traceback.print_exc()
    raw_input()
                """.format(
                    acquisitionStr=acquisitionStr,
                    minQual=metadata['minimum_correlation:'],
                    minCalZ=metadata['minimum_z_calibration:'],
                    maxCalZ=metadata['maximum_z_calibration:'],
                    nm_per_pixel=(metadata['nanometers_per_x_pixel:'],
                                  metadata['nanometers_per_y_pixel:'],
                                  metadata['nanometers_per_z_pixel:']),
                    nm_per_bin=metadata['nanometers_per_histogram_bin:'],
                    minX=metadata['minimum_x:'],
                    maxX=metadata['maximum_x:'],
                    minY=metadata['minimum_y:'],
                    maxY=metadata['maximum_y:'],
                    minZ=metadata['minimum_z:'],
                    maxZ=metadata['maximum_z:'],
                    linkedInput=str(linkedInput)
                    ).replace('\n', os.linesep))
            ipython_run('plots.py')
            return None

##    def inspection(self, fol):
##        print fol
##        return None

def human_sorted(myList):
    """ Sort the given list in the way that humans expect.
    """
    def convert(text):
        return int(text) if text.isdigit() else text
    def alphanum_key(key):
        return [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(myList, key=alphanum_key)


## Go!
root = tk.Tk()
gui = Gui(root)
root.mainloop()

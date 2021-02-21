# The following function is used to return _f for the latest output
def latest_output(path):
    """
    This function looks in the dump_f folder in the provided path
    directory and identifies the latest dump file.
    WARNING this function is currently unsafe for corrupted dump files, which 
    can occur in particular if an allocation expires while the code is writing
    to disk.
    """
    import numpy as np
    import glob
    import os
    dump_f_names = np.sort(glob.glob(os.path.join(path, 'dump_f/t=*.bin')))

    ndumps = dump_f_names.size

    if(ndumps > 0):
        latest_f = dump_f_names[ndumps-1].rsplit('.',1)[0]
        time_elapsed = float(latest_f.rsplit('/',1)[1][2:])
        return latest_f, time_elapsed
    else:
        return None, None

def format_time(time):
    """
    This function formats a time value given as an integer or a float and
    returns a string appropriately padded with zeros.
    Right now format is [8].[6].
    """
    x = float(time)
    return (str(x).split('.')[0].zfill(8) + '.'
          + str(x).split('.')[1].ljust(6, '0'))

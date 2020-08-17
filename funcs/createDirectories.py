""" Create directories to save the relevant files for a given file """
import os
import csv
def createDirectories(folder, filename):
    dirName = filename
    try:
        # Create target Directory
        os.mkdir(folder+'/'+filename)
        print("Directory ", dirName,  " Created ")
    except FileExistsError:
        print("Directory ", dirName,  " already exists")

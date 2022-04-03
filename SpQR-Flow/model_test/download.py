""" Module with download_file function """
import os
import sys
import logging

import requests


def download_file(url, filename):
    """
    Function handling download of the file object
    """

    # check if file already exists
    if os.path.isfile(filename):
        print(f'File {filename} exists.')
        return

    # create directory and download the file
    try:
        os.makedirs(os.path.dirname(filename))
    except FileExistsError:
        pass
    # check if URL is valid
    try:
        response = requests.get(url, stream=True)
    except requests.URLRequired:
        logging.exception('Insert valid URL')
        return

    with open(filename, 'wb') as newfile:
        step_size = 1000000
        file_size = response.header['Content-length']
        print(f'File size: {file_size * 1e6 : .3f} MB \n')
        for step, chunk in enumerate(
                    response.iter_content(chunk_size=step_size)):
            newfile.write(chunk)
            sys.stdout.flush()
            progress = (step * step_size * 100 / file_size)
            sys.stdout.write(f'Download progress:{progress:.0f}% \r')
    if os.path.isfile(filename):
        print(f'File {filename} downloaded succesfully')
    else:
        print('The file could not be downloaded :( \nTry to download it manually.')
    return

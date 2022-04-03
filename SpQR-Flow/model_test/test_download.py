""" Unit tests for download.py module """
import os
import unittest

from PIL import Image
from requests.exceptions import URLRequired, MissingSchema, InvalidSchema, InvalidURL

from download import download_file


class TestDownload(unittest.TestCase):
    """ Tests for download.py module """
    def test_invalid_url(self):
        """
        Testing IsValidURL function
        """
        errors = (URLRequired, MissingSchema, InvalidSchema, InvalidURL)
        self.assertRaises(errors, download_file, 'an invalid url','test.txt')

    def test_download(self):
        """
        Testing for a succesfull download
        """
        url = 'https://cds.cern.ch/images/CMS-PHO-EVENTS-2012-005-1/file?size=original'
        outfile = os.path.join(os.path.dirname(__file__),
                               'download/Hgammagamma.png')
        download_file(url, outfile)
        self.assertTrue(os.path.isfile(outfile))
        image = Image.open(outfile)
        image.show()


if __name__ == '__main__':
    unittest.main()

import os
import sys

import requests

def download_data(data_url,
	path_to_data=os.path.join(
		os.path.dirname(os.path.abspath(__file__)),
		'download/HIGGS.csv.gz'),
	verbose=True):

	if os.path.isfile(path_to_data):
		if verbose:
			print(f'File {path_to_data} already exists')
		return

	os.mkdir(os.path.dirname(path_to_data))
	response = requests.get(data_url, stream=True)
	with open(path_to_data, 'wb') as file_data:
		tot_length = int(response.headers.get('content-length'))
		step_size = 1000000
		for step, chunk in enumerate(
				response.iter_content(chunk_size=step_size)):
			file_data.write(chunk)
			sys.stdout.flush()
			progress = step * step_size * 100 / tot_length
			sys.stdout.write(f'Download progress: {progress:.1f}% \r')
	if os.path.isfile(path_to_data):
		print(f'File {path_to_data} has been downloaded succesfully')

data_url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00280/HIGGS.csv.gz'
download_dir = os.path.dirname(os.path.abspath(__file__))
download_dir = os.path.join(download_dir, 'download')
path_to_data = os.path.join(download_dir, 'HIGGS.csv.gz')

download_data(data_url, path_to_data=path_to_data)

 

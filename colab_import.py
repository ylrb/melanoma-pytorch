import os
import zipfile

REPO_URL = 'https://github.com/IgrecL/melanoma-pytorch/archive/master.zip'
ARCHIVE_NAME = 'cvml-pytorch-master.zip'
EXTRACTION_DIR = '/content'

os.chdir('/content')
!wget {REPO_URL} -O {ARCHIVE_NAME}
os.makedirs(EXTRACTION_DIR, exist_ok=True)

with zipfile.ZipFile(ARCHIVE_NAME, 'r') as zip_ref:
    zip_ref.extractall(EXTRACTION_DIR)

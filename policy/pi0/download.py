import os
from openpi.shared import download

os.environ['OPENPI_DATA_HOME'] = '/mnt/data/baijun/UniVTAC/policy/pi0/openpi'
path = download.maybe_download(
    "gs://openpi-assets/checkpoints/pi0_fast_base/params"
)
print(path)
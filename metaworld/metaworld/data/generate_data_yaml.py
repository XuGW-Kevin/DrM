# get all the data_paths and save to a yaml

import subprocess

import yaml

command = "azcopy list 'https://rlnexusstorage2.blob.core.windows.net/00-share-data-public/metaworld/' | sed 's/Content Length: .*//g' | sed 's/INFO: //g' | sed 's/azcopy: A newer version 10.18.1 is available to download//g'"
process = subprocess.Popen(command, shell=True,stdout=subprocess.PIPE)
output, error = process.communicate()
data_paths = output.decode('utf-8').replace('\n','').split(';  ')
data_paths = data_paths[:-1] # Remove the last empty path.

data_paths_dict = {}
for path in data_paths:
    # data_path = path.replace('/root/.xt/mnt/data_container/', '')
    path = 'metaworld/' + path
    if "224x224" in path:
        env_name = path.split('metaworld/224x224/corner/')[1].split('/Sawyer')[0]
        noise = path.split('/Sawyer/')[1].split('/')[0]
        data_name = env_name + '-' + noise + '-224x224'
    else:
        env_name =  path.split('metaworld/')[1].split('/Sawyer')[0]
        noise = path.split('/Sawyer/')[1].split('/')[0]
        data_name = env_name + '-' + noise

    data_paths_dict[data_name] = path

import yaml
with open('data_paths.yaml', 'w') as file:
    yaml.dump(data_paths_dict, file)
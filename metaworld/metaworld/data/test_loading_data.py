from metaworld import get_mw_env_and_data

import yaml
with open('data_paths.yaml', 'r') as file:
    data_paths = yaml.safe_load(file)

for k in data_paths:
    print('Loading data', k)
    get_mw_env_and_data(k)

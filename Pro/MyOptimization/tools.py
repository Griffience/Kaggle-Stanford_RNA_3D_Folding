import yaml


#加载配置文件
class Config:
    def __init__(self, **entries):
        self.__dict__.update(entries)
        self.entries = entries
    def print(self):
        print(self.entries)

def load_config_from_yaml(file_path):
    with open(file_path, 'r') as file:
        conf = yaml.safe_load(file)
    return Config(**conf)
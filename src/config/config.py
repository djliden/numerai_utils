from os import PathLike
import yaml

class Configuration:
    def __init__(self,
                 default_cfg:PathLike,
                 model_cfg:PathLike,
                 user_cfg:PathLike = None):
        """Class for accessing and updating configuration files

        kwargs:
        * default_cfg: path to default configutation file
        * model_cfg: path to model-default configuration file
        * user_cfg: path to user-specified configuration file
        """
        self.def = yaml.load(open(default_cfg, 'r'), Loader = yaml.CLoader)
        self.model = yaml.load(open(model_cfg, 'r'), Loader = yaml.CLoader)
        if user_cfg:
            self.user = yaml.load(open(user_cfg, 'r'), Loader = yaml.CLoader)

    def merge_configs(self):
        self.config = dict()
        self.config.update(self.def)
        self.config.update(self.model)
        if user_cfg: self.config.update(self.user)

    def init_config(self):
        self.merge_configs()
        self.__dict__.update(self.config)
    
class Log:
    def __init__(self, config):
        self.config = config
    
    

from os.path import splitext
import yaml
import textwrap

class BaseConfig:
    __isFrozen = True
    def __init__(self):
        self.__isFrozen = True
    def __setattr__(self, key, value):
        """Prevent directly setting attributes

        This is to ensure that all configuration options are defined
        directly through the dictionary or YAML adding methods. This
        limitation could be lifted in the future.
        """
        if self.__isFrozen and not hasattr(self, key):
            raise TypeError(textwrap.fill(textwrap.dedent("""
            Directly adding new attributes to a Configuration 
            object using dot notation is not supported. Please use 
            the update_config() method""")))
        object.__setattr__(self, key, value)
        
    def _freeze(self):
        """Prevent changes to the __dict__"""
        self.__isFrozen = True
        
    def _thaw(self):
        """Allow changes to the __dict__"""
        self.__isFrozen = False
    
    def _update_state(self):
        for key in self.config.keys():
            if isinstance(self.config[key], dict):
                self.__setattr__(key, Config(self.config[key]))

    def merge_dicts(self, dict1, dict2):
        """ Recursively merges dict2 into dict1
        source: https://stackoverflow.com/a/24837438/11598548
        
        """
        if not isinstance(dict1, dict) or not isinstance(dict2, dict):
            return dict2
        for k in dict2:
            if k in dict1:
                dict1[k] = self.merge_dicts(dict1[k], dict2[k])
            else:
                dict1[k] = dict2[k]
        return dict1           

    def update_from_yaml(self, new_config):
        """Update the stored configuration

        kwargs:
        * new_config: path to YAML file

        If new_config includes a key/value pair already included in the
        existing config, the existing config will be overwritten with
        the new value. If new_config includes new key/value pairs, they
        will be added to the existing config.
        """
        new_config = yaml.load(open(new_config, 'r'), Loader = yaml.SafeLoader)
        self.config = self.merge_dicts(self.config, new_config)
        self.__dict__.update(self.config)
        
    def update_from_dict(self, new_config:dict):
        """Update the stored configuration

        kwargs:
        * new_config: python dict

        If new_config includes a key/value pair already included in the
        existing config, the existing config will be overwritten with
        the new value. If new_config includes new key/value pairs, they
        will be added to the existing config.
        """
        self.config = self.merge_dicts(self.config, new_config)
        self.__dict__.update(new_config)
        

class Config(BaseConfig):
    def __init__(self, config):
        #self._isFrozen = True
        super().__init__()
        self._thaw()
        self.config = dict()
        self.update_config(config)
        self._freeze()
        
    def update_config(self, new_config):
        self._thaw()
        if isinstance(new_config, dict):
            self.update_from_dict(new_config)
        elif splitext(new_config)[1]=='.yaml':
            self.update_from_yaml(new_config)
        else:
            raise TypeError(textwrap.fill(textwrap.dedent("""
            Please pass either a dictionary or the path to a YAML file 
            with a .yaml extension as an argument to Config(). Other
            configuration formats are not supported.
            """)))
        self._update_state()
        self._freeze()

    def dump_config(self, path=None):
        """Print or save YAML version of config file
        
        kwargs:
        * path (default None):  if non-null, path to which to save output
        """
        if path:
            stream = open(path, 'w')
            yaml.dump(self.config, stream)
        print(yaml.dump(self.config, Dumper=yaml.SafeDumper))
        
    def __repr__(self):
        return f'Config Object with Keys:\n{yaml.dump(self.config, Dumper=yaml.SafeDumper)}'
    def __str__(self):
        return str(yaml.dump(self.config, Dumper=yaml.SafeDumper))

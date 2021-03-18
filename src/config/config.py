import yaml
import textwrap

class Config():
    __isFrozen = True
    def __init__(self, config):
        self._thaw()
        self.config = dict()
        if isinstance(config, dict):
            self.update_from_dict(config)
        else:
            self.update_from_yaml(config)
        self._update_state()
        self._freeze()
        
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

                
    def update_config(self, new_config):
        self._thaw()
        if isinstance(new_config, dict):
            self.update_from_dict(new_config)
        else:
            self.update_from_yaml(new_config)
        self._update_state()
        self._freeze()
        
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
        self.config.update(new_config)
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
        self.config.update(new_config)    
        self.__dict__.update(new_config)

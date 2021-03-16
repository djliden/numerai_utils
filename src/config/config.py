from os import PathLike
from fastcore.basics import AttrDict
import yaml

"""
An important note: At this phase, the classes contained here muddy the
relationship between the class instance's __dict__ and the stored
configuration dictionaries (e.g. Configuration.config). This could cause
problems if one were to try to *update* the configuration by defining a
class attribute, e.g. Configuration.Model.lr = 0.1. I see the most potential for this to be problematic in the log output -- anything added in this way would
not be included in the .config or .log dictionary and thus would not be
included in the output. I hope to find a good way to make both approaches
work, but in the meantime, to be safe, it is better to *add* to the config
only using the functions designed for that purpose. Accessing parts of the
config through class attributes should currently be acceptable, though.

Maybe something from here would help https://docs.python.org/3/reference/datamodel.html?emulating-container-types#emulating-container-types

Or here https://stackoverflow.com/questions/38034377/object-like-attribute-access-for-nested-dictionary

From SE:

class AttrDict(dict):
"" Dictionary subclass whose entries can be accessed by attributes
    (as well as normally).
''
def __init__(self, *args, **kwargs):
    def from_nested_dict(data):
        "" Construct nested AttrDicts from nested dictionaries. ""
        if not isinstance(data, dict):
            return data
        else:
            return AttrDict({key: from_nested_dict(data[key])
                                for key in data})

    super(AttrDict, self).__init__(*args, **kwargs)
    self.__dict__ = self

    for key in self.keys():
        self[key] = from_nested_dict(self[key])

Or I can just copy YACS https://github.com/rbgirshick/yacs/blob/32d5e4ac300eca6cd3b839097dde39c4017a1070/yacs/config.py#L112

The challenge is with class attribute <-> dict interoperability.

"""


class Container(dict):
    """A parent class for configuration and logging classes"""
    def __init__(self, *args, **kwargs):
        super(Container, self).__init__(*args, **kwargs)
        self.__dict__ = self
        
    @staticmethod
    def from_nested_dicts(data):
        """ Construct nested AttrDicts from nested dictionaries."""
        if not isinstance(data, dict):
            return data
        else:
            return Container({key: Container.from_nested_dicts(data[key])
                                for key in data})
        
    def _freeze(self):
        """Prevent changes to the __dict__"""
        self.__isFrozen = True
        
    def _thaw(self):
        """Allow changes to the __dict__"""
        self.__isFrozen = False

    
class Configuration(Container):
    """Class for accessing and updating configuration files
    
        kwargs:
        * config: path to an initial configuration file
        """
    __isFrozen = False
    
    def __init__(self, config = None):
        super().__init__()
        self.config = dict()
        if config:
            config = yaml.load(open(config, 'r'), Loader = yaml.SafeLoader)
            self.config.update(config)
        new_dict = self.from_nested_dicts(self.config)
        #self.__dict__.update(self.config)
        self.__dict__ = new_dict
        self._freeze(self)
        
        

    def __setattr__(self, key, value):
        """Prevent directly setting attributes

        This is to ensure that all configuration options are defined
        directly through the dictionary or YAML adding methods. This
        limitation could be lifted in the future.
        """
        if self.__isFrozen and not hasattr(self, key):
            raise TypeError(("Directly adding new attributes to a Configuration "
                             "object using dot notation is not supported. Please use "
                             "th update_from_yaml or update_from_dict methods"))
        object.__setattr__(self, key, value)


    
        
    def update_from_yaml(self, new_config):
        """Update the stored configuration

        kwargs:
        * new_config: path to YAML file

        If new_config includes a key/value pair already included in the
        existing config, the existing config will be overwritten with
        the new value. If new_config includes new key/value pairs, they
        will be added to the existing config.
        """
        self._thaw()
        new_config = yaml.load(open(new_config, 'r'), Loader = yaml.SafeLoader)
        self.config.update(new_config)
        self.__dict__.update(self.config)
        self._freeze()
        
    def update_from_dict(self, new_config:dict):
        """Update the stored configuration

        kwargs:
        * new_config: python dict

        If new_config includes a key/value pair already included in the
        existing config, the existing config will be overwritten with
        the new value. If new_config includes new key/value pairs, they
        will be added to the existing config.
        """
        self._thaw()
        self.config.update(new_config)    
        self.__dict__.update(self.config)
        self._freeze()

    def print_config(self):
        print(yaml.dump(self.config))

    
class Log:
    """Class for logging and returning output of model runs

    kwargs:
    * include_config: whether the configuration should be included in
      the log output
    * config (dict): the configuration to be included, if include_config
      is True. Note that this argument expects a dictionary, so from an
      instance of the Configuration class, pass Configuration.config to
      the log class.
    """
    def __init__(self, config:dict, incude_config:bool = True):
        self.log = dict()
        if include_config:
            self.config = config
            self.log.update(self.config)
        self.__dict__.update(self.log)
    
    
    def update_from_dict(self, new_log:dict):
        """Update the stored configuration

        kwargs:
        * new_log: python dict

        If new_log includes a key/value pair already included in the
        existing log, the existing log will be overwritten with
        the new value. If new_log includes new key/value pairs, they
        will be added to the existing log.
        """
        self.log.update(new_log)    
        self.__dict__.update(self.log)

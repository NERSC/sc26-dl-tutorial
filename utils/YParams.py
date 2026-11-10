"""YAML configuration object with one shared parameter dictionary."""

from ruamel.yaml import YAML
import logging


class YParams:
    """Load one named config so attributes and items read the same dictionary."""

    def __init__(self, yaml_filename, config_name, print_params=False):
        object.__setattr__(self, "_yaml_filename", yaml_filename)
        object.__setattr__(self, "_config_name", config_name)
        object.__setattr__(self, "params", {})

        with open(yaml_filename) as _file:
            for key, val in YAML().load(_file)[config_name].items():
                if val == "None":
                    val = None

                self.params[key] = val

        if print_params:
            self.log()

    def __getattr__(self, key):
        try:
            return self.params[key]
        except KeyError:
            raise AttributeError(key) from None

    def __setattr__(self, key, val):
        if key.startswith("_") or key == "params":
            object.__setattr__(self, key, val)
        else:
            self.params[key] = val

    def __getitem__(self, key):
        return self.params[key]

    def __setitem__(self, key, val):
        self.params[key] = val

    def get(self, key, default=None):
        """Return a parameter, or ``default`` when it is absent."""
        return self.params.get(key, default)

    def log(self):
        """Log the loaded configuration."""
        logging.info("------------------ Configuration ------------------")
        logging.info("Configuration file: " + str(self._yaml_filename))
        logging.info("Configuration name: " + str(self._config_name))
        for key, val in self.params.items():
            logging.info(str(key) + " " + str(val))
        logging.info("---------------------------------------------------")

    def update(self, new_params):
        """Merge ``new_params`` into the parameter dictionary."""
        self.params.update(new_params)

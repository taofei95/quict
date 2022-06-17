import os
import yaml


def _get_default_config():
    """ Generate the dict of the default config for the simulator.

    Returns:
        [dict]: The default config dict for simulator.
    """
    curPath = os.path.dirname(os.path.realpath(__file__))
    simPath = os.path.split(curPath)[0]
    confPath = os.path.join(simPath, "config", "default.yml")

    with open(confPath, 'r', encoding='utf-8') as f:
        config = f.read()

    config = yaml.load(config, Loader=yaml.FullLoader)

    return config


_DEFAULT_CONFIG = _get_default_config()


def option_validation():
    """ Check options' correctness for the given simulator. """
    def decorator(func):
        def wraps(self, **kwargs):
            device = self._device
            backend = self._backend
            if device in ["GPU", "CPU"]:
                default_options = _DEFAULT_CONFIG[device][backend]
            else:
                default_options = _DEFAULT_CONFIG[device]

            customized_options = kwargs["options"]
            if customized_options.keys() - default_options.keys():
                raise KeyError(f"There are some unsupportted options in current simulator. {customized_options}")
            else:
                return func(self, customized_options, default_options)

        return wraps

    return decorator

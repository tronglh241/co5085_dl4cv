from __future__ import annotations

import copy
from importlib import import_module
from pathlib import Path
from typing import Any, Union

import yaml
from yacs.config import _VALID_TYPES
from yacs.config import CfgNode as _CfgNode
from yacs.config import _assert_with_logging, _valid_type

# Special keys used for dynamic configuration evaluation
EXTRALIBS = 'extralibs'
MODULE = 'module'
NAME = 'name'
KWARGS = 'kwargs'


class CfgNode(_CfgNode):
    '''
    A subclass of yacs.config.CfgNode that adds dynamic evaluation features.
    '''

    @classmethod
    def load(
        cls,
        filepath: Union[str, Path],
        evaluate: bool = True,
    ) -> CfgNode:
        '''
        Loads a configuration from a YAML file and optionally evaluates it.

        Args:
            filepath (str or Path): Path to the YAML configuration file.
            evaluate (bool): Whether to evaluate dynamic elements in the configuration.

        Returns:
            CfgNode: A CfgNode instance with the loaded configuration.
        '''
        # Open and read the YAML file, loading it as a configuration object
        with open(filepath, 'r') as f:
            config = CfgNode.load_cfg(f)

        # Convert dictionary-like elements recursively to CfgNode instances
        config = cls._convert_to_cfg_node(config)

        if evaluate:
            # Evaluate dynamic expressions
            config = config.eval()

        # Ensure the resulting object is of type CfgNode
        assert isinstance(config, CfgNode)
        return config

    @classmethod
    def _convert_to_cfg_node(cls, data: Any) -> Any:
        '''
        Recursively converts dictionary-like objects to instances of CfgNode.

        Args:
            data (Any): Data to be converted.

        Returns:
            Any: Converted data as CfgNode or the same type if not applicable.
        '''
        if isinstance(data, dict):
            # Convert dictionary to CfgNode and recursively convert its values
            cfg_node = cls()
            for key, value in data.items():
                cfg_node[key] = cls._convert_to_cfg_node(value)
            return cfg_node
        elif isinstance(data, list):
            # Recursively convert list elements
            return [cls._convert_to_cfg_node(item) for item in data]
        else:
            # Return the data unchanged if it is not a dict or list
            return data

    @classmethod
    def _create_config_tree_from_dict(cls, dic: dict, key_list: list) -> dict:
        '''
        Create a configuration tree using the given dict.
        Any dict-like objects inside dict will be treated as a new CfgNode.

        Args:
            dic (dict):
            key_list (list[str]): a list of names which index this CfgNode from the root.
                Currently only used for logging purposes.
        '''
        dic = copy.deepcopy(dic)
        for k, v in dic.items():
            if isinstance(v, dict):
                # Convert dict to CfgNode
                dic[k] = cls(v, key_list=key_list + [k])
            elif isinstance(v, (list, tuple)):
                dic[k] = type(v)(cls(ele) if isinstance(ele, dict) else ele for ele in v)
        return dic

    def __setattr__(self, name: str, value: Any) -> None:
        if self.is_frozen():
            raise AttributeError(
                'Attempted to set {} to {}, but CfgNode is immutable'.format(
                    name, value
                )
            )

        _assert_with_logging(
            name not in self.__dict__,
            'Invalid attempt to modify internal CfgNode state: {}'.format(name),
        )

        self[name] = self._convert_to_cfg_node(value)

    def dump(self, **kwargs: Any) -> str:
        '''Dump to a string.'''

        def convert_to_dict(cfg_node: Any, key_list: list) -> Any:
            if not isinstance(cfg_node, CfgNode):
                _assert_with_logging(
                    _valid_type(cfg_node),
                    'Key {} with value {} is not a valid type to dump; valid types: {}'.format(
                        '.'.join(key_list), type(cfg_node), _VALID_TYPES
                    ),
                )
                return cfg_node
            else:
                cfg_dict = dict(cfg_node)
                for k, v in cfg_dict.items():
                    cfg_dict[k] = convert_to_dict(v, key_list + [k])
                return cfg_dict

        self_as_dict = convert_to_dict(self, [])
        return yaml.safe_dump(self_as_dict, **kwargs)

    def eval(self) -> CfgNode:
        '''
        Evaluates the configuration to resolve dynamic elements such as modules and objects.

        Returns:
            CfgNode: A new evaluated configuration.
        '''
        config = org_config = self.clone()
        extralibs = {}

        # Process the external libraries defined in the configuration
        for alias, lib_info in config.pop(EXTRALIBS, {}).items():
            if isinstance(lib_info, dict):
                # Import a specific object from a module
                module = lib_info.pop(MODULE, None)
                name = lib_info.pop(NAME, None)

                if module is None or name is None:
                    raise ValueError(
                        f'`{MODULE}` and `{NAME}` must be both specified in `{EXTRALIBS}`, '
                        f'found {MODULE}: {module}, {NAME}: {name}.'
                    )

                if lib_info:
                    raise ValueError(f'Only {MODULE} and {NAME} are allowed, found {[key for key in lib_info.keys()]}.')

                lib = getattr(import_module(module), name)
            else:
                # Import the entire module
                lib = import_module(lib_info)

            extralibs[alias] = lib  # Store the imported library

        # Evaluate the configuration with the imported libraries
        config = CfgNode._eval(config, extralibs, org_config)

        return config

    @staticmethod
    def _eval(config: Any, global_context: dict, local_context: dict) -> Any:
        '''
        Recursively evaluates the configuration to resolve dynamic values.

        Args:
            config (Any): Configuration object to evaluate.
            global_context (dict): Global context for evaluation (e.g., imported modules).
            local_context (dict): Local context for evaluation (e.g., configuration itself).

        Returns:
            Any: Evaluated configuration object.
        '''
        if isinstance(config, dict):
            initialized = (
                len(config) == 2 and MODULE in config and NAME in config
                or len(config) == 3 and MODULE in config and NAME in config and KWARGS in config
            )

            if initialized:
                # Process dictionary configuration items
                module = config.pop(MODULE, None)
                name = config.pop(NAME, None)

            # Recursively evaluate dictionary values
            for key, value in config.items():
                config[key] = CfgNode._eval(value, global_context, local_context)

            # If both module and name are specified, construct the object with kwargs
            if initialized:
                kwargs = config.pop(KWARGS, {})  # Extract arguments for the callable
                config = eval(name, {}, vars(import_module(module)))(**kwargs)
            elif not isinstance(config, CfgNode):
                # Convert the dict back to a CfgNode if it is not already
                config = CfgNode(config)

        elif isinstance(config, (list, tuple)):
            # Recursively evaluate list or tuple items
            config = type(config)(map(lambda ele: CfgNode._eval(ele, global_context, local_context), config))

        elif isinstance(config, str):
            # Evaluate strings as expressions
            config = eval(config, global_context, local_context)

            # If the result is not a string, evaluate it further
            if not isinstance(config, str):
                config = CfgNode._eval(config, global_context, local_context)

        return config

"""MultiModalModule is a LightningModule that allows switching between multiple model modes."""

import functools
from collections.abc import Callable
from typing import Any, ClassVar

import lightning as L  # noqa: N812


class UnknownModelModeError(ValueError):
    """Exception raised when the model mode is unknown."""

    def __init__(self, mode: str) -> None:
        """Initializes the UnknownModelModeError.

        Parameters
        ----------
        mode : str
            The mode that caused the error.

        """
        self.mode = mode
        super().__init__(f"Unknown model mode: {mode}")


class MultiModalModule(L.LightningModule):
    """MultiModalModule is a LightningModule that allows multiple model modes.

    Parameters
    ----------
    mode : str
        Model mode.

    Attributes
    ----------
    mode : str
        Model mode.

    Raises
    ------
    UnknownModelModeError
        If the model mode is unknown.

    """

    modes: ClassVar[tuple[str]] = []

    def __init__(self, mode: str) -> None:
        """Initializes the MultiModalModule.

        Parameters
        ----------
        mode : str
            Model mode.

        """
        super().__init__()
        self.mode = mode

    @property
    def mode(self) -> str:
        """Gets the current model mode.

        Returns
        -------
        str
            The current model mode.

        """
        return self._mode

    @mode.setter
    def mode(self, mode: str) -> None:
        """Sets the current model mode.

        Parameters
        ----------
        mode : str
            The model mode to set.

        Raises
        ------
        UnknownModelModeError
            If the mode is unknown.

        """
        if mode not in self.modes:
            raise UnknownModelModeError(mode)
        self._mode = mode

    def __init_subclass__(cls, modes: list[str] | tuple[str] = ()) -> None:
        """Initializes a subclass with the given modes.

        Parameters
        ----------
        cls : type
            The subclass to initialize.
        modes : Optional[list[str]]
            The modes to set for the subclass. If None, an empty list is used.

        """
        super().__init_subclass__()
        cls.modes = tuple(modes)
        # Find all methods with dispatch information
        dispatch_methods = {}

        for attr_name in dir(cls):
            attr = getattr(cls, attr_name)
            if hasattr(attr, "_dispatch_mode") and hasattr(attr, "_dispatch_function"):
                mode = attr._dispatch_mode  # noqa: SLF001
                func_name = attr._dispatch_function  # noqa: SLF001

                if func_name not in dispatch_methods:
                    dispatch_methods[func_name] = {}

                dispatch_methods[func_name][mode] = attr

        # Create dispatcher methods
        for func_name, method_modes in dispatch_methods.items():
            dispatcher = cls._create_dispatcher(method_modes)
            setattr(cls, func_name, dispatcher)

    @classmethod
    def _create_dispatcher(cls, func_name: str, funcs_by_mode: dict[str, Callable]) -> Callable:
        """Creates a dispatcher method that calls the correct function based on the current mode.

        Parameters
        ----------
        func_name : str
            The name of the function to dispatch.
        funcs_by_mode : dict[str, Callable]
            A dictionary where the keys are the model modes and the values are the functions to be
                called for that mode.

        Returns
        -------
        Callable
            A dispatcher method that will call the correct function based on the current model mode.

        Raises
        ------
        UnknownModelModeError
            If the current model mode is not in the funcs_by_mode dictionary.

        """

        def dispatcher(self: MultiModalModule, *args: list[Any], **kwargs: dict[str, Any]) -> Any:  # noqa: ANN401
            current_mode = self.mode
            if current_mode not in funcs_by_mode:
                raise UnknownModelModeError(current_mode)

            return funcs_by_mode[current_mode](self, *args, **kwargs)

        if hasattr(cls, func_name):
            return functools.wraps(getattr(cls, func_name))(dispatcher)
        return dispatcher


def bind(mode: str, function: str) -> Callable:
    """Binds a function to a specific mode.

    This decorator is used to specify that a function should be called when a
    specific mode is set. The function will only be called if the current mode
    matches the mode specified.

    Parameters
    ----------
    mode : str
        The mode to bind to.
    function : str
        The name of the function to bind.

    Returns
    -------
    Callable
        A wrapper function that calls the original function if the current mode
        matches the specified mode.

    """

    def _wrapper(func: Callable) -> Callable:
        func._dispatch_mode = mode  # noqa: SLF001
        func._dispatch_function = function  # noqa: SLF001
        return func

    return _wrapper

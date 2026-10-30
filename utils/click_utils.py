"""Utils for click-based console apps."""

import re

import click


class IntOrFloatParamType(click.ParamType):
    """A custom click parameter type that converts a given value to an int or float.

    If the value is already an int or float, it is returned as is.
    If the value is a string, it is converted to a float.
    If the string contains a decimal point, the float is returned.
    If the string does not contain a decimal point, the string is converted to an int.

    If the value cannot be converted to an int or float, a ValueError is raised.

    :param value: The value to convert.
    :param param: The click parameter that triggered this conversion.
    :param ctx: The click context that triggered this conversion.
    :return: The converted value.
    :raises ValueError: If the value cannot be converted to an int or float.
    """

    name = "int or float"

    def convert(self, value: str, param: click.Parameter, ctx: click.Context) -> int | float:
        """Converts the given value to an int or float.

        If the value is already an int or float, it is returned as is.
        If the value is a string, it is converted to a float.
        If the string contains a decimal point, the float is returned.
        If the string does not contain a decimal point, the string is converted to an int.

        If the value cannot be converted to an int or float, a ValueError is raised.

        :param value: The value to convert.
        :param param: The click parameter that triggered this conversion.
        :param ctx: The click context that triggered this conversion.
        :return: The converted value.
        :raises ValueError: If the value cannot be converted to an int or float.
        """
        if isinstance(value, (int, float)):
            return value
        try:
            value_str = str(value)
            value = float(value_str)
            if re.search(r"\.", value_str):
                return value
            return int(value)
        except ValueError:
            self.fail(f"{value} is not a valid int or float", param, ctx)


INT_OR_FLOAT = IntOrFloatParamType()

import json
from pathlib import Path
import numpy as np
from typing import Dict, Any, List, Tuple, Union

class ColourTempCurve:
    """
    Class that maps colour temperature to red and blue values using a piecewise linear function
    in 2 dimensions.
    """
    def __init__(self, points : List[float]) -> None:
        """
        Initialise a 2D piecewise linear function from a list of points. The points come in groups of three,
        representing a colour temperature and a pair of red and blue values.

        Args:
            points (List[float]): List of points in the format [colour_temp1, red1, blue1, colour_temp2, red2, blue2, ...]
        """
        self.colour_temp = np.array(points[0::3])
        self.red = np.array(points[1::3])
        self.blue = np.array(points[2::3])

    def eval(self, colour_temp : float) -> np.ndarray:
        """
        Evaluate the function at a given colour temperature to get red and blue values.

        Args:
            colour_temp (float): Colour temperature in Kelvin

        Returns:
            np.ndarray: Array of red and blue values
        """
        red = np.interp(colour_temp, self.colour_temp, self.red)
        blue = np.interp(colour_temp, self.colour_temp, self.blue)
        return np.array([red, blue])

    def invert(self, red_blue : np.ndarray) -> float:
        """
        Invert the function to find the colour temperature for a given pair of r and b values.

        Args:
            red_blue (np.ndarray): Array of red and blue values

        Returns:
            float: Colour temperature in Kelvin
        """
        # This isn't precise, but should be close enough.
        red, blue = red_blue
        colour_temp_red = np.interp(red, self.red[::-1], self.colour_temp[::-1])
        colour_temp_blue = np.interp(blue, self.blue, self.colour_temp)
        return (colour_temp_red + colour_temp_blue) / 2

class Tuning:
    """
    A class to hold a Raspberry Pi cameratuning file. Raspberry Pi 5 style tuning files should
    be used.
    """

    def __init__(self, json_config : str) -> None:
        """
        Initialise a tuning from a JSON configuration.

        Args:
            json_config (str): Path to the JSON configuration file
        """
        self.json_config = json_config

        self.colour_temp_curve = ColourTempCurve(self.get_algorithm("awb")["ct_curve"])

    @staticmethod
    def load(json_file : Union[str, Path]) -> 'Tuning':
        """
        Load the tuning from a JSON file.

        Args:
            json_file (Union[str, Path]): Path to the JSON configuration file

        Returns:
            Tuning: The tuning object
        """
        with open(json_file, 'r') as f:
            config = json.load(f)
        return Tuning(config)

    SEARCH_PATH = [Path("."), Path(__file__).resolve().parent / "tunings"]

    @staticmethod
    def find(sensor : str) -> Union[str, Path]:
        """
        Find the tuning for a given sensor, checking the folders listed in SEARCH_PATH.

        Args:
            sensor (str): Sensor model name

        Returns:
            Union[str, Path]: Path to the tuning file
        """
        for path in Tuning.SEARCH_PATH:
            tuning_file = path / f"{sensor}.json"
            if tuning_file.exists():
                return tuning_file

        raise FileNotFoundError(f"Tuning file for {sensor} not found")

    def get_algorithm(self, name: str) -> Dict[str, Any]:
        """
        Get algorithm configuration by name.
        Searches the 'algorithms' list for a dictionary with a key that ends with the given name.

        Args:
            name (str): Name of the algorithm to find

        Returns:
            Dict[str, Any]: The algorithm configuration dictionary

        Raises:
            KeyError: If the algorithm is not found
        """
        if "algorithms" not in self.json_config:
            raise KeyError("No 'algorithms' section found in configuration")

        algorithms = self.json_config["algorithms"]
        for algorithm in algorithms:
            for key, value in algorithm.items():
                if key.endswith(f".{name}"):
                    return value

        raise KeyError(f"Algorithm '{name}' not found in configuration")

    def get_colour_values(self, colour_temp: float) -> np.ndarray:
        """
        Get the colour values for a given colour temperature.

        Args:
            colour_temp (float): Colour temperature in Kelvin

        Returns:
            np.ndarray: Array of red and blue values
        """
        return self.colour_temp_curve.eval(colour_temp)

    def get_colour_temp(self, red_blue : np.ndarray) -> float:
        """
        Get the colour temperature for a given pair of red and blue values.

        Args:
            red_blue (np.ndarray): Array of red and blue values

        Returns:
            float: Colour temperature in Kelvin
        """
        return self.colour_temp_curve.invert(red_blue)

    def get_black_level(self, bits : int = 16) -> int:
        """
        Get the black level for the camera.

        Args:
            bits (int): Number of bits in the final black level output value (default 16)

        Returns:
            int: The black level
        """
        # The value in the file is always in 16 bits.
        return self.get_algorithm("black_level")["black_level"] >> (16 - bits)

    @staticmethod
    def _interpolate_table(colour_temp: float, tables : List[Dict[str, Any]]) -> np.ndarray:
        """
        Interpolate a table for a given colour temperature.
        """
        if colour_temp <= tables[0]["ct"]:
            return np.array(tables[0]["table"]).reshape(32, 32)
        elif colour_temp >= tables[-1]["ct"]:
            return np.array(tables[-1]["table"]).reshape(32, 32)

        # Find the two tables that bracket the given colour temperature
        for table, next_table in zip(tables[:-1], tables[1:]):
            if table["ct"] <= colour_temp and next_table["ct"] >= colour_temp:
                alpha = (colour_temp - table["ct"]) / (next_table["ct"] - table["ct"])
                return (alpha * np.array(next_table["table"]) + (1 - alpha) * np.array(table["table"])).reshape(32, 32)

        raise RuntimeError("Internal error: failed to interpolate LSC tables - should not happen")

    def get_lsc_tables(self, colour_temp: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Get LSC tables for a given colour temperature.

        Args:
            colour_temp (float): Colour temperature in Kelvin

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: The LSC tables for the R, G, and B channels
        """
        lsc_config = self.get_algorithm("alsc")
        cr_table = Tuning._interpolate_table(colour_temp, lsc_config["calibrations_Cr"])
        cb_table = Tuning._interpolate_table(colour_temp, lsc_config["calibrations_Cb"])
        luminance_table = np.array(lsc_config["luminance_lut"]).reshape(32, 32)
        luminance_strength = lsc_config.get("luminance_strength", 1.0)
        luminance_table = (luminance_table - 1.0) * luminance_strength + 1.0

        r_table = cr_table * luminance_table
        g_table = luminance_table
        b_table = cb_table * luminance_table

        return r_table / r_table.min(), g_table / g_table.min(), b_table / b_table.min()

    def get_gamma_curve(self) -> Tuple[List[int], List[int]]:
        """
        Get the gamma curve for the camera. Return two lists, one for the x values and one for the y values.
        """
        contrast_config = self.get_algorithm("contrast")
        gamma_curve = contrast_config["gamma_curve"]
        return (gamma_curve[0::2], gamma_curve[1::2])

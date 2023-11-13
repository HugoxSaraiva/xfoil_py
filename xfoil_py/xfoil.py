import os
import re
import sys
import logging
import itertools
import concurrent.futures
import numpy as np
import subprocess as sp
from .utils import utils
from scipy.special import comb
from contextlib import suppress
from matplotlib import pyplot as plt
from .utils.utils import log
from .definitions import EXEC_DIR
from .utils.user_options import UserOptions
# TODO: use BezierSegment from matplotlib


class XFoil:
    DEFAULT_OPTIONS = {
        "panels": 300,
        "max_iterations": 100,
        "n_crit": 9,
        "max_threads": 6,
        "executable_path": EXEC_DIR,
        "save_polar_name": None,
        "process_timeout": 45,
        "disable_graphics": False
    }

    @log("Initializing XFoil class", logging.INFO)
    def __init__(self, name, mach, reynolds, alphas, **koptions):
        """
        Initializes XFOIL class with given parameters. All inputs can be lists defining multiple airfoils or test cases.
        :param name: Name of the airfoil, can be a valid naca name or a path to a dat file.
        :param mach: Mach number to run xfoil executable on.
        :param reynolds: Reynolds number to run xfoil executable on, must be an Int
        :param alphas: List of floats corresponding to [alpha_min, alpha_max, alpha_step].
        :param koptions: Keyword arguments for other xfoil executable running options.
        """
        # Initial guard clauses
        alphas = np.array(alphas, dtype=float).reshape(-1)
        if len(alphas) % 3 != 0:
            raise IndexError(
                f"{str(self.__class__.__name__)}.alphas length is not divisible by 3."
                f"Can't identify min, max and step values"
            )

        # Get user input options and set defaults
        self._options = {**koptions}
        for key in self.DEFAULT_OPTIONS:
            if self._options.get(key, None) is None:
                self._options[key] = self.DEFAULT_OPTIONS[key]

        # Attributes to use with xfoil executable
        self.names = np.array(name, dtype=str, ndmin=1)
        self.machs = np.array(mach, dtype=float, ndmin=1)
        self.reynolds = np.array(reynolds, dtype=int, ndmin=1)
        self.alphas = alphas.reshape(-1, 3)
        self.save_polar_name = utils.add_prefix_suffix(self._options.get("save_polar_name", None), suffix=".txt")

        # Additional parameters available to users to use with xfoil executable
        self.panels = np.array(self._options["panels"], dtype=int, ndmin=1)
        self.n_crit = np.array(self._options["n_crit"], dtype=int, ndmin=1)
        self.max_iterations = np.array(self._options["max_iterations"], dtype=int, ndmin=1)

        # Attributes to use in XFoil class
        self._executable = self._options["executable_path"]
        self._process_timeout = self._options["process_timeout"]
        self._max_threads = self._options["max_threads"]
        self._disable_graphics = self._options["disable_graphics"]
        self.results = {}

        cases_to_iterate = [
            self.machs,
            self.reynolds,
            self.alphas,
            self.panels,
            self.n_crit,
            self.max_iterations
        ]
        fill_values = [
            # Repeats last value given if the array is to small
            self.machs[-1],
            self.reynolds[-1],
            self.alphas[-1],
            self.panels[-1],
            self.n_crit[-1],
            self.max_iterations[-1]
        ]
        # Create iterator for all test cases
        test_cases_iterator = utils.zip_longest_modified(*cases_to_iterate, fillvalue=fill_values)
        # Create iterator for all airfoils and test cases
        self._airfoils_iterator = itertools.product(self.names, test_cases_iterator)
        logging.debug(f"XFoil class initialized: {repr(self)}")

    def __repr__(self):
        """
        Returns string that can be used to instantiate current class object
        :return: string
        """
        template = "{class_name}({arguments})"
        kwargs_string = ', '.join(
            '{0}={1!r}'.format(key, value)
            for key, value in self._options.items()
            if value != self.DEFAULT_OPTIONS.get(key, None)
        )
        function_arguments = [
            np.array2string(self.names, separator=","),
            np.array2string(self.machs, separator=","),
            np.array2string(self.reynolds, separator=","),
            np.array2string(self.alphas, separator=",")
        ]
        if kwargs_string:
            function_arguments.append(kwargs_string)

        arguments_string = ", ".join(function_arguments)

        formatted_string = template.format(
            class_name=str(self.__class__.__name__),
            arguments=arguments_string
        )
        return formatted_string

    @log("XFoil class run() method called", logging.INFO)
    def run(self):
        """
        Runs xfoil executable with given parameters used to initialize class. Results are then stored in self.results
        :return: None
        """
        constants = {
            "save_name": self.save_polar_name,
            "disable_graphics": self._disable_graphics,
            "executable_path": self._executable,
            "timeout": self._process_timeout
        }
        with concurrent.futures.ThreadPoolExecutor(max_workers=self._max_threads) as executor:
            # Create dict to identify results as they are completed
            futures_to_run_id = {}
            for run_id, (name, test_case) in enumerate(self._airfoils_iterator):
                mach, reynolds, alphas, panels, n_crit, max_iterations = test_case
                run_data = {
                    "run_id": run_id,
                    "name": name,
                    "mach": mach,
                    "reynolds": reynolds,
                    "alphas": alphas,
                    "panels": panels,
                    "n_crit": n_crit,
                    "max_iterations": max_iterations,
                    "disable_graphics": self._disable_graphics,
                }
                run_data.update(constants)
                # XFoil._worker_run is the method that actually runs the xfoil executable
                future = executor.submit(XFoil._worker_run, **run_data)
                futures_to_run_id[future] = run_id

            # Loop through completed subprocesses and append them to self.results on corresponding run_id
            for future in concurrent.futures.as_completed(futures_to_run_id):
                run_id = futures_to_run_id[future]
                self.results[run_id] = future.result()
                logging.info(f"Run of id {run_id} finished")

    @classmethod
    def from_dict(cls, dictionary):
        """
        Creates a class instance from a dictionary.
        :param dictionary: dict
        :return: class
        """
        return cls(**dictionary)

    @classmethod
    def from_text_file(cls, file_path):
        """
        Creates a class instance from a text file. Each key must be in a new line and in the format KEY=VALUE.
        :param file_path: path to text file
        :return: class
        """
        # Creating dict from file
        with open(file_path) as file:
            dictionary = {}
            for line in file:
                key_string, value_string = line.strip().split("=")
                key = key_string.strip()
                if key == "save_polar_name" or key == "executable_path":
                    values = value_string.strip()
                else:
                    values = [value for value in value_string.strip().split(" ")]
                dictionary[key] = values
        return XFoil.from_dict(dictionary)

    @classmethod
    def from_arguments(cls, *args):
        """
        Creates class instance from input arguments.
        :param args:
        :return:
        """
        arguments = UserOptions()
        arguments.parse_arguments(*args)
        # Sanitizing which args are passed to instantiate class
        dictionary = {
            'name': arguments.name,
            'mach': arguments.mach,
            'reynolds': arguments.reynolds,
            'alphas': arguments.alphas,
            'save_polar_name': arguments.save_polar_name,
            'executable_path': arguments.executable_path,
            'max_threads': arguments.max_threads
        }
        return XFoil.from_dict(dictionary)

    @staticmethod
    def _worker_run(**run_data):
        """
        Runs xfoil executable with given parameters
        :return: Dict with result_data
        """
        data = run_data.copy()
        # Getting all variables from dict
        keys = ["name", "mach", "reynolds", "alphas", "panels", "n_crit", "max_iterations", "save_name"]
        misc_keys = ["run_id", "disable_graphics", "executable_path", "timeout", "save_name"]
        run_id, disable_graphics, executable_path, timeout, save_name = tuple(map(data.get, misc_keys))

        result_data = {'metadata': {key: data.get(key) for key in keys}}
        logging.debug(f"XFoil class _worker_run() method called with parameters: {data}")

        # Creating temporary filename
        tmp_file_name = utils.add_prefix_suffix(utils.random_string(), prefix="tmp_", suffix=".txt")
        data.update({"polar_name": tmp_file_name})

        # Create input_string to xfoil executable subprocess
        input_string = XFoil._generate_input_string(data)

        logging.info("Opening subprocess")
        # Runs xfoil subprocess
        try:
            sp.run(executable_path,
                   input=input_string.encode(),
                   capture_output=True,
                   check=True,
                   timeout=timeout)
        except FileNotFoundError:
            raise ExecutableNotFoundError(f"Executable {executable_path} not found")
        except TypeError as e:
            if executable_path is None:
                raise ExecutableNotFoundError(f"Executable is None")
            raise e
        except TimeoutError:
            logging.warning("Process killed due to timeout")

        # Subprocess finished, proceeding to read file contents:
        logging.info("Proceeding to read polar file")
        try:
            result_data['result'] = XFoil.read_polar(tmp_file_name)
        except FileNotFoundError:
            logging.warning("An error occurred while trying to read polar")
            logging.debug(f"Tried to read polar {tmp_file_name} and failed")
            result_data['result'] = None
        finally:
            # Clean tmp files created by xfoil executable
            if save_name:
                XFoil._file_cleanup(tmp_file_name, f"{run_id}_{save_name}")
            else:
                XFoil._file_cleanup(tmp_file_name)
            logging.debug(f"Data is {result_data}")
            logging.info("XFoil class _worker_run() method ended")
            return result_data

    @staticmethod
    @log("File cleanup called", logging.INFO)
    def _file_cleanup(filepath, save_name=None):
        """
        Cleans files generated by xfoil executable and renames tmp file if save_name is not None
        :param filepath: string: tmp file path
        :param save_name: string: file save_name
        :return: None
        """
        if save_name and os.path.exists(filepath):
            # Delete file if it exists already and then rename file
            with suppress(OSError):
                os.remove(save_name)
            os.rename(filepath, save_name)
            logging.info(f"Keeping polar file {save_name} on disk")
        elif utils.path_leaf(filepath).startswith("tmp_"):
            logging.debug(f"Deleting polar file {filepath}")
            with suppress(OSError):
                os.remove(filepath)

        # Deleting additional file created on linux
        with suppress(OSError):
            os.remove(":00.bl")

    @staticmethod
    def _is_naca(string):
        """
        Gets an input string and returns true if it is a 4 or 5 digit naca number and returns false otherwise
        :param string: string
        :return: boolean
        """
        return string.isdigit() and (len(string) in [4, 5])

    @staticmethod
    @log("Generating input string", logging.INFO)
    def _generate_input_string(data_dict):
        """
        Method that generates the input string for xfoil executable, it emulates real user inputs on xfoil
        executable.
        :param data_dict: data with xfoil executable parameters
        :return: string
        """
        input_string = []
        # Check whether to use naca from xfoil or load a dat file
        if XFoil._is_naca(data_dict["name"]):
            input_string.append(f"naca {data_dict['name']}\n")
        else:
            input_string.append(f"load {utils.add_prefix_suffix(data_dict['name'], suffix='.dat')}\n")
            input_string.append(f"name {utils.path_leaf(data_dict['name'])}\n")

        # Disabling airfoil plotter from appearing
        if data_dict["disable_graphics"]:
            print("Disabling graphics")
            input_string.append("plop\n")
            input_string.append("g 0\n")
            input_string.append("\n")

        # Setting number of panels
        input_string.append("ppar\n")
        input_string.append(f"n {data_dict['panels']}\n\n\n")

        # Setting viscous calculation parameters
        input_string.append("oper\n")
        input_string.append(f"visc {data_dict['reynolds']}\n")
        input_string.append(f"M {data_dict['mach']}\n")
        input_string.append(f"iter {data_dict['max_iterations']}\n")
        input_string.append("vpar\n")
        input_string.append(f"n {data_dict['n_crit']}\n\n")

        # Setting polar name
        input_string.append("pacc\n")
        input_string.append(f"{data_dict['polar_name']}\n\n")

        # Setting alpha values
        input_string.append("aseq\n")
        input_string.append(f"{data_dict['alphas'][0]}\n")
        input_string.append(f"{data_dict['alphas'][1]}\n")
        input_string.append(f"{data_dict['alphas'][2]}\n")
        input_string.append("\nquit\n")

        logging.debug(f"Input string is : {input_string}")
        return "".join(input_string)

    @staticmethod
    def _bezier_curve_2d(control_points):
        """
        Returns a function that calculates the x,y values of the corresponding bezier curve given the control points
        :param control_points: list of control points
        :return: Bezier function for given control points
        """
        n = len(control_points) - 1
        control_points_array = np.array(control_points)
        return lambda t: sum(
            comb(n, i) * (1 - t) ** (n - i) * t ** i * p for i, p in enumerate(control_points_array)
        )

    @staticmethod
    def read_dat(file_path):
        """
        Method that reads a .dat file containing x,y coordinates of an airfoil in Selig format
        :param file_path: String containing file to read
        :return: dict with "x" and "y" arrays
        """
        # Certifies that the .dat suffix is in file_path
        file_path = utils.add_prefix_suffix(file_path, suffix=".dat")
        logging.debug(f"Reading dat file {file_path}")

        regex = re.compile('((?:\s[-\s]?)\d\.\d+)')
        with open(file_path, 'r') as f:
            lines = f.readlines()
            x = []
            y = []
            for line in lines:
                linedata = regex.findall(line)
                if linedata:
                    try:
                        x.append(float(linedata[0]))
                        y.append(float(linedata[1]))
                    except IndexError:
                        raise InvalidFileContentsError(f"No valid coordinates found in file: '{file_path}'")

        dat_contents = {'x': np.array(x), 'y': np.array(y)}
        if any([len(values) == 0 for _, values in dat_contents.items()]):
            raise InvalidFileContentsError(f"No valid coordinates found in file: '{file_path}'")
        return dat_contents

    @staticmethod
    def read_polar(file_path):
        """
        Static method to read xfoil polar from txt given its file name. Returns a dict with airfoil polar results
        :param file_path: string
        :return: dict with polar file data
        """
        # Certifies that the .txt suffix is in file_path
        file_path = utils.add_prefix_suffix(file_path, suffix=".txt")
        logging.debug(f"Reading polar file {file_path}")

        data_regex = re.compile('([+-]?\d+\.\d+)')
        columns_regex = re.compile('(\w+)')

        with open(file_path) as f:
            lines = f.readlines()
            columns = columns_regex.findall(lines[10])
            logging.debug(f"Columns are {columns}")
            polar_contents = {}
            for column in columns:
                polar_contents[column] = []

            for line in lines[12:]:
                line_data = data_regex.findall(line)
                if line_data:
                    for i, column in enumerate(columns):
                        try:
                            polar_contents[column].append(float(line_data[i]))
                        except IndexError:
                            raise InvalidFileContentsError(f"No valid polar found in file: '{file_path}'")

            if any([len(values) == 0 for _, values in polar_contents.items()]):
                raise InvalidFileContentsError(f"No valid polar found in file: '{file_path}'")
            logging.debug(f"Polar contents are: {polar_contents}")
            return polar_contents

    @staticmethod
    def create_airfoil(save_file_name, upper_control_points, lower_control_points):
        """
        Receives 2 2D arrays (x,y coordinates of control points) and creates a .dat file to use with xfoil
        Overwrites file if it exists
        :param save_file_name: name of the file to save
        :param upper_control_points: array of upper control points
        :param lower_control_points: array of lower control points
        :return: 2D array of airfoil coordinates
        """
        # Certifies file has extension
        save_file_name = utils.add_prefix_suffix(save_file_name, suffix=".dat")

        logging.debug(f"Creating airfoil dat file with name: {save_file_name}")

        upper_control_points = np.array(upper_control_points).reshape((-1, 2))
        lower_control_points = np.array(lower_control_points).reshape((-1, 2))
        upper_curve = XFoil._bezier_curve_2d(upper_control_points)
        lower_curve = XFoil._bezier_curve_2d(lower_control_points)

        # Generate varying x increments for better resolution around trailing and leading edges
        dt = np.linspace(0, 1, 80)
        dx = (1 - np.cos(2 * dt * np.pi / 2)) / 2

        # Generate x,y points to save onto .dat file
        xy_upper = np.empty([len(dx), 2])
        xy_lower = np.empty([len(dx), 2])
        for i, x in enumerate(dx):
            xy_upper[i] = upper_curve(x)
            xy_lower[i] = lower_curve(x)
        xy_upper_reversed = xy_upper[::-1]

        dat_xy = np.concatenate((xy_upper_reversed[:-2], xy_lower[1:]))
        logging.debug(f"XY to save coordinates is: \n{dat_xy}")

        # Save points to .dat file
        with open(save_file_name, 'w') as f:
            for x, y in dat_xy:
                f.write("  {:.6f} {: .6f}\n".format(x, y))
        logging.debug("Dat file saved")
        return dat_xy

    @staticmethod
    def delete_airfoil(file_path):
        """
        Deletes .dat file of airfoil
        :param file_path: string with airfoil path
        :return: None
        """
        # Certifies that the .dat suffix is in file_path
        file_path = utils.add_prefix_suffix(file_path, suffix=".dat")
        logging.debug(f"Deleting file: {file_path}")
        with suppress(OSError):
            os.remove(file_path)

    @staticmethod
    @log("Plotting airfoil", logging.INFO)
    def plot_airfoil(file_path, separate_curves=False):
        """
        Plots airfoil using matplotlib
        :param file_path: path of the airfoil dat file
        :param separate_curves: Boolean to indicate if user wants upper and lower surfaces with different colors or not
        :return: None
        """
        # No need to sanitize file_path since read_dat does it already
        data = XFoil.read_dat(file_path)

        if separate_curves:
            x_upper, x_lower = np.array_split(data['x'], 2)
            y_upper, y_lower = np.array_split(data['y'], 2)

            # Repeat point at x = 0
            x_lower = np.insert(x_lower, 0, x_upper[-1])
            y_lower = np.insert(y_lower, 0, y_upper[-1])

            plt.plot(x_upper, y_upper)
            plt.plot(x_lower, y_lower)
        else:
            plt.plot(data['x'], data['y'])
        plt.show()


class InvalidFileContentsError(Exception):
    pass


class ExecutableNotFoundError(Exception):
    pass


def main(arguments):
    """
    Runs XFoil class from given arguments
    :param arguments: arguments to be parsed
    :return: results dict or None if "show" is True
    """
    args = UserOptions()
    args.parse_arguments(arguments)
    log_level = logging.DEBUG if args.debug else logging.WARNING
    logging.basicConfig(
        level=log_level,
        stream=sys.stdout
    )
    logging.getLogger("matplotlib.font_manager").setLevel(logging.WARNING)

    if args.show:
        XFoil.plot_airfoil(args.name)
        return

    xfoil_obj = XFoil.from_arguments(arguments)
    xfoil_obj.run()
    return xfoil_obj.results


if __name__ == "__main__":
    main(sys.argv[1:])

import concurrent.futures
import numpy as np
import os
import re
import subprocess as sp
import sys
from contextlib import suppress
from matplotlib import pyplot as plt
from scipy.special import comb
from xfoil_py.utils.user_options import UserOptions
from xfoil_py.utils.utils import *
from xfoil_py.definitions import EXEC_DIR


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
        "disable_graphics": True
    }

    @log("Initializing XFoil class", logging.INFO)
    def __init__(
            self,
            name,
            mach,
            reynolds,
            alphas,
            **koptions
    ):
        # Initial guard clauses
        alphas = np.array(alphas, dtype=float).reshape(-1)
        if len(alphas) % 3 != 0:
            raise IndexError(
                "alphas length is not divisible by 3. Can't identify min, max and step values"
            )

        # Get options and set defaults
        options_dict = koptions.copy()
        for key in self.DEFAULT_OPTIONS:
            if options_dict.get(key, None) is None:
                options_dict[key] = self.DEFAULT_OPTIONS[key]

        # Attributes to use with xfoil
        self.names = np.array(name, dtype=str, ndmin=1)
        self.machs = np.array(mach, dtype=float, ndmin=1)
        self.reynolds = np.array(reynolds, dtype=int, ndmin=1)
        self.alphas = alphas.reshape(-1, 3)

        save_polar_name = options_dict.get("save_polar_name", None)
        self.save_polar_name = add_prefix_suffix(save_polar_name, suffix=".txt") if save_polar_name else None

        # Additional parameters available to users
        self.panels = np.array(options_dict["panels"], dtype=int, ndmin=1)
        self.n_crit = np.array(options_dict["n_crit"], dtype=int, ndmin=1)
        self.max_iterations = np.array(options_dict["max_iterations"], dtype=int, ndmin=1)

        cases_to_iterate = [
            self.machs,
            self.reynolds,
            self.alphas,
            self.panels,
            self.n_crit,
            self.max_iterations
        ]
        fill_values = [
            self.machs[-1],
            self.reynolds[-1],
            self.alphas[-1],
            self.panels[-1],
            self.n_crit[-1],
            self.max_iterations[-1]
        ]
        # Create iterator for all test cases
        test_cases_iterator = zip_longest_modified(*cases_to_iterate, fillvalue=fill_values)
        # Create iterator for all airfoils and test cases
        self._airfoils_iterator = itertools.product(self.names, test_cases_iterator)

        # Attributes to use in XFoil class
        self._executable = options_dict["executable_path"]
        self._process_timeout = options_dict["process_timeout"]
        self._max_threads = options_dict["max_threads"]
        self._disable_graphics = options_dict["disable_graphics"]
        self._options = options_dict
        self.results = {}

        debug_msg = "XFoil class initialized with properties: names: {} machs: {} reynolds: {} alphas: {}"
        logging.debug(debug_msg.format(self.names,
                                       self.machs,
                                       self.reynolds,
                                       self.alphas
                                       )
                      )

    def __repr__(self):
        template = "{}({}, {}, {}, {}, {})"
        kwargs_string = ', '.join('{0}={1!r}'.format(key, value) for key, value in self._options.items())
        formatted_string = template.format(
            str(self.__class__.__name__),
            np.array2string(self.names, separator=","),
            np.array2string(self.machs, separator=","),
            np.array2string(self.reynolds, separator=","),
            np.array2string(self.alphas, separator=","),
            kwargs_string
        )

        return formatted_string

    @log("XFoil class run() method called", logging.INFO)
    def run(self):
        """
        Runs xfoil with given filename, mach, reynolds, alpha_min, alpha_max and alpha_step, used to initialize class
        :return: None
        """
        xfoil_run_constants = (self.save_polar_name, self._disable_graphics, self._executable, self._process_timeout)
        with concurrent.futures.ThreadPoolExecutor(max_workers=self._max_threads) as executor:
            # Create dict to identify results as they are completed
            futures_to_run_id = {}
            for run_id, (name, test_case) in enumerate(self._airfoils_iterator):
                futures_to_run_id[executor.submit(XFoil._worker_run,
                                                  run_id,
                                                  name,
                                                  *test_case,
                                                  *xfoil_run_constants)] = run_id
            for future in concurrent.futures.as_completed(futures_to_run_id):
                run_id = futures_to_run_id[future]
                self.results[run_id] = future.result()
                logging.info(f"Run of id {run_id} finished")

    @classmethod
    def from_dict(cls, dictionary):
        return cls(**dictionary)

    @classmethod
    def from_text_file(cls, file_path):
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
    def from_arguments(cls, arguments):
        """
        Creates class instance from input string
        """
        args = UserOptions()
        args.parse_arguments(arguments)
        # Can't iterate over args attributes because it has more than what is used to instantiate class
        dictionary = {
            'name': args.name,
            'mach': args.mach,
            'reynolds': args.reynolds,
            'alphas': args.alphas,
            'save_polar_name': args.save_polar_name,
            'executable_path': args.executable_path,
            'max_threads': args.max_threads
        }
        return XFoil.from_dict(dictionary)

    @staticmethod
    def _worker_run(run_id, name, mach, reynolds, alphas, panels, n_crit, max_iterations, save_name, disable_graphics,
                    executable_path, timeout):
        """
        Runs xfoil executable with given parameters
        :return: Dict with result_data
        """
        result_data = {'metadata': {'name': name,
                                    'mach': mach,
                                    'reynolds': reynolds,
                                    'alphas': alphas,
                                    'panels': panels,
                                    'n_crit': n_crit,
                                    'max_iterations': max_iterations}}
        logging.debug(f"XFoil class _worker_run() method called with parameters: {result_data}"
                      f"disable_graphics:{disable_graphics}, executable_path{executable_path}, timeout:{timeout}")

        # Creating temporary filename
        tmp_file_name = add_prefix_suffix(random_string(), prefix="tmp_", suffix=".txt")
        logging.info("Opening subprocess")
        logging.debug(f"Executable path is {executable_path}")

        data_dict = {
            "name": name,
            "mach": mach,
            "reynolds": reynolds,
            "alphas": alphas,
            "panels": panels,
            "n_crit": n_crit,
            "max_iterations": max_iterations,
            "polar_name": tmp_file_name,
            "disable_graphics": disable_graphics
        }
        input_string = XFoil._generate_input_string(data_dict)

        logging.info("Communicating input string to subprocess")

        # Runs xfoil subprocess
        try:
            process = sp.run(executable_path,
                             input="".join(input_string).encode(),
                             capture_output=True,
                             check=True,
                             timeout=timeout)
        except FileNotFoundError:
            raise ExecutableNotFoundError(f"Executable {executable_path} not found")
        except TimeoutError:
            logging.warning("Process killed due to timeout")

        try:
            logging.info("Proceeding to read polar file")
            # Reading polar text file to get result_data
            result_data['result'] = XFoil.read_polar(tmp_file_name)
        except FileNotFoundError:
            logging.warning("An error occurred while trying to read polar")
            logging.debug(f"Tried to read polar {tmp_file_name} and failed")
            result_data = None
        finally:
            if save_name:
                XFoil._file_cleanup(tmp_file_name, f"{run_id}_{save_name}")
            else:
                XFoil._file_cleanup(tmp_file_name)
            logging.debug(f"Data is {result_data}")
            logging.info("XFoil class run() method ended")
            return result_data

    @staticmethod
    @log("File cleanup called", logging.INFO)
    def _file_cleanup(filepath, save_name=None):
        if save_name and os.path.exists(filepath):
            # Delete file if it exists already and then rename file
            with suppress(OSError):
                os.remove(save_name)
            os.rename(filepath, save_name)
            logging.info(f"Keeping polar file {save_name} on disk")
        elif path_leaf(filepath).startswith("tmp_"):
            logging.debug(f"Deleting polar file {filepath}")
            with suppress(OSError):
                os.remove(filepath)

        # Deleting additional file created on linux
        with suppress(OSError):
            os.remove(":00.bl")

    # def __read_stdout(self):
    #     logging.info("Reading stdout")
    #     exp = "((?<=x[/]c\s=\s{2})\d\.\d*|\d+(?=\s{3}rms)|" \
    #           "(?<=(?:Cm|CL|CD)\s=\s)[\s-]?\d*\.\d*|" \
    #           "(?<=a\s=\s)[\s-]?\d*\.\d*|" \
    #           "(?<=CDp\s=\s)[\s-]?\d*\.\d*)"
    #     regex = re.compile(exp)
    #
    #     data = regex.findall(self._stdout)
    #     a = []
    #     cl = []
    #     cd = []
    #     cdp = []
    #     cm = []
    #     xtr_top = []
    #     xtr_bottom = []
    #
    #     # Pick data from only the last iteration for each angle
    #     i = 0
    #     while i < len(data):
    #         iteration = float(data[i + 2])
    #         if i + 10 > len(data) or iteration > float(data[i + 10]):
    #             xtr_top.append(float(data[i]))
    #             xtr_bottom.append(float(data[i + 1]))
    #             a.append(float(data[i + 3]))
    #             cl.append(float(data[i + 4]))
    #             cm.append(float(data[i + 5]))
    #             cd.append(float(data[i + 6]))
    #             cdp.append(float(data[i + 7]))
    #         i = i + 8
    #     return {'a': np.array(a), 'cl': np.array(cl), 'cd': np.array(cd), 'cdp': np.array(cdp), 'cm': np.array(cm),
    #             'xtr_top': np.array(xtr_top), 'xtr_bottom': np.array(xtr_bottom)}

    @staticmethod
    def _is_naca(string):
        """
        Gets an input string and returns true if it is a 4 or 5 digit naca number and returns false if it isn't
        :param string: string
        :return: boolean
        """
        return string.isdigit() and (len(string) in [4, 5])

    @staticmethod
    @log("Generating input string", logging.INFO)
    def _generate_input_string(data_dict):
        input_string = []
        # Check whether to use naca from xfoil or load a dat file
        if XFoil._is_naca(data_dict["name"]):
            input_string.append(f"naca {data_dict['name']}\n")
        else:
            input_string.append(f"load {add_prefix_suffix(data_dict['name'], suffix='.dat')}\n")
            input_string.append(f"{path_leaf(data_dict['name'])}\n")

        # Disabling airfoil plotter from appearing
        if data_dict["disable_graphics"]:
            input_string.append("plop\n")
            input_string.append("g 0\n\n")

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
        return input_string

    @staticmethod
    def _bezier_curve_2d(control_points):
        """
        Returns a function that calculates the x,y values of the corresponding bezier curve given the control points
        :param control_points: list of control points
        :return: function
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
        file_path = add_prefix_suffix(file_path, suffix=".dat")
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
        Static method to read xfoil polar from txt given its file name. Returns a dict with airfoil polar split up
        :param file_path: string
        :return: dict with polar file data
        """
        # Certifies that the .txt suffix is in file_path
        file_path = add_prefix_suffix(file_path, suffix=".txt")
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
        save_file_name = add_prefix_suffix(save_file_name, suffix=".dat")

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
        # Certifies that the .dat suffix is in file_path
        file_path = add_prefix_suffix(file_path, suffix=".dat")
        logging.debug(f"Deleting file: {file_path}")
        with suppress(OSError):
            os.remove(file_path)

    @staticmethod
    @log("Plotting airfoil", logging.INFO)
    def plot_airfoil(file_path, separate_curves=False):
        """
        Plots airfoil using matplotlib
        :param file_path: path of the airfoil dat file
        :param separate_curves:
        :return:
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

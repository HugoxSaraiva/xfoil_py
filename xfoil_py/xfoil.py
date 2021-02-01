import concurrent.futures
import logging
import numpy as np
import os
import re
import subprocess as sp
import sys
from matplotlib import pyplot as plt
from scipy.special import comb
from xfoil_py.utils.user_options import UserOptions
from xfoil_py.utils.utils import *
from xfoil_py.definitions import EXEC_DIR
# TODO: use BezierSegment from matplotlib
# TODO: Add from_file and from dict methods to instantiate class


class XFoil:
    def __init__(
            self,
            name,
            mach,
            reynolds,
            alphas,
            save_polar_name=None,
            executable_path=None,
            panels=300,
            max_iterations=100,
            n_crit=9,
            max_threads=None
    ):
        logging.info("Initializing XFoil class")
        # Attributes to use with xfoil
        self.names = np.array(name, dtype=str, ndmin=1)
        self.save_polar_name = add_prefix_suffix(save_polar_name, suffix=".txt") if save_polar_name else None
        self.machs = np.array(mach, dtype=float, ndmin=1)
        self.reynolds = np.array(reynolds, dtype=int, ndmin=1)
        if len(alphas) % 3 == 0:
            self.alphas = np.array(alphas, dtype=float).reshape(-1, 3)
        else:
            raise IndexError("alphas length is not divisible by 3. Can't identify min, max and step values")

        # Additional parameters available to users
        self.panels = np.array(panels, dtype=int, ndmin=1)
        self.n_crit = np.array(n_crit, dtype=int, ndmin=1)
        self.max_iterations = np.array(max_iterations, dtype=int, ndmin=1)
        # Create iterator for all test cases
        test_cases_iterator = zip_longest_modified(self.machs,
                                                   self.reynolds,
                                                   self.alphas,
                                                   self.panels,
                                                   self.n_crit,
                                                   self.max_iterations,
                                                   fillvalue=[self.machs[-1],
                                                              self.reynolds[-1],
                                                              self.alphas[-1],
                                                              self.panels[-1],
                                                              self.n_crit[-1],
                                                              self.max_iterations[-1]
                                                              ]
                                                   )
        # Create iterator for all airfoils and test cases
        self.airfoils_iterator = itertools.product(self.names, test_cases_iterator)

        # Attributes to use in XFoil class
        self._executable = [executable_path if executable_path else EXEC_DIR]
        self._process_timeout = 30
        self.max_threads = max_threads
        self._disable_graphics = True
        self.results = {}

        debug_msg = "XFoil class initialized with properties: names: {} machs: {} reynolds: {} alphas: {}"
        logging.debug(debug_msg.format(self.names,
                                       self.machs,
                                       self.reynolds,
                                       self.alphas
                                       )
                      )

    def run(self):
        """
        Runs xfoil with given filename, mach, reynolds, alpha_min, alpha_max and alpha_step, used to initialize class
        :return: None
        """
        logging.info("XFoil class run() method called")
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_threads) as executor:
            # Create dict to identify results as they are completed
            futures_to_run_id = {
                executor.submit(XFoil._worker_run,
                                        name,
                                        mach,
                                        reynolds,
                                        alphas,
                                        panels,
                                        n_crit,
                                        self.save_polar_name,
                                        run_id,
                                        iterations,
                                        self._disable_graphics,
                                        self._executable,
                                        self._process_timeout): run_id
                for run_id, (name,
                             (mach, reynolds, alphas, panels, n_crit, iterations)
                             ) in enumerate(self.airfoils_iterator)
            }
            for future in concurrent.futures.as_completed(futures_to_run_id):
                run_id = futures_to_run_id[future]
                self.results[str(run_id)] = future.result()
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
    def _worker_run(name,
                    mach,
                    reynolds,
                    alphas,
                    panels,
                    n_crit,
                    save_name,
                    run_id,
                    max_iterations,
                    disable_graphics,
                    executable_path,
                    timeout
                    ):
        """
        Runs xfoil executable with given parameters
        :return: Dict with data
        """
        data = {'metadata': {'name': name,
                             'mach': mach,
                             'reynolds': reynolds,
                             'alphas': alphas,
                             'panels': panels,
                             'n_crit': n_crit,
                             'max_iterations': max_iterations}}
        logging.debug(f"XFoil class _worker_run() method called with parameters: {data}"
                      f"disable_graphics:{disable_graphics}, executable_path{executable_path}, timeout:{timeout}")

        # Creating temporary filename
        tmp_file_name = add_prefix_suffix(random_string(), prefix="tmp_", suffix=".txt")
        logging.info("Opening subprocess")
        logging.debug(f"Executable path is {executable_path}")

        input_string = XFoil._generate_input_string(name,
                                                    mach,
                                                    reynolds,
                                                    alphas,
                                                    panels,
                                                    n_crit,
                                                    max_iterations,
                                                    tmp_file_name,
                                                    disable_graphics)

        logging.info("Communicating input string to subprocess")
        # Runs xfoil subprocess
        try:
            completed_process = sp.run(executable_path,
                                       input="".join(input_string).encode(),
                                       capture_output=True,
                                       check=True,
                                       timeout=timeout)
            # if logging.root.level <= logging.DEBUG:
            #     stdout = completed_process.stdout.decode()
            #     stderr = completed_process.stderr.decode()
            #     logging.debug(f"Stdout is: {stdout}")
            #     logging.debug(f"Error is: {stderr}")
        except FileNotFoundError:
            raise ExecutableNotFoundError(f"Executable {executable_path} not found")
        except TimeoutError:
            logging.warning("Process killed due to timeout")

        try:
            logging.info("Proceeding to read polar file")
            # Reading polar text file to get data
            data['result'] = XFoil.read_polar(tmp_file_name)
        except FileNotFoundError:
            logging.warning("An error occurred while trying to read polar")
            logging.debug(f"Tried to read polar {tmp_file_name} and failed")
            data = None
        finally:
            if save_name:
                XFoil._file_cleanup(tmp_file_name, f"{run_id}_{save_name}")
            else:
                XFoil._file_cleanup(tmp_file_name)
            logging.debug(f"Data is {data}")
            logging.info("XFoil class run() method ended")
            return data

    # def plot_polar(self, x_column_name, y_column_name, save_plot_name=None, plot_title=None):
    #     logging.info("Plotting polar")
    #     if not self.results:
    #         raise TypeError("XFoil result is None")
    #     plt.plot(x_column_name, y_column_name, data=self.results)
    #     plt.xlabel(x_column_name)
    #     plt.ylabel(y_column_name)
    #     plt.title(plot_title)
    #
    #     # Stop blocking code execution when debugging
    #     if not self.debug:
    #         plt.show()
    #     if save_plot_name:
    #         plt.savefig(save_plot_name)

    @staticmethod
    def _file_cleanup(filepath, save_name=None):
        if save_name and os.path.exists(filepath):
            # Delete file if it exists already and then rename file
            logging.info(f"Keeping polar file {save_name} on disk")
            if os.path.exists(save_name):
                os.remove(save_name)
            os.rename(filepath, save_name)
        elif path_leaf(filepath).startswith("tmp_"):
            logging.debug(f"Deleting polar file {filepath}")
            if os.path.exists(filepath):
                os.remove(filepath)

        # Deleting additional file created on linux
        if os.path.exists(":00.bl"):
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
    def _generate_input_string(name, mach, reynolds, alphas, panels, n_crit, iterations, polar_name, disable_graphics):
        logging.info("Creating input string")
        input_string = []
        # Check whether to use naca from xfoil or load a dat file
        if XFoil._is_naca(name):
            input_string.append(f"naca {name}\n")
        else:
            input_string.append(f"load {add_prefix_suffix(name, suffix='.dat')}\n")
            input_string.append(f"{path_leaf(name)}\n")

        # Disabling airfoil plotter from appearing
        if disable_graphics:
            input_string.append("plop\n")
            input_string.append("g 0\n\n")

        # Setting number of panels
        input_string.append("ppar\n")
        input_string.append(f"n {panels}\n\n\n")

        # Setting viscous calculation parameters
        input_string.append("oper\n")
        input_string.append(f"visc {reynolds}\n")
        input_string.append(f"M {mach}\n")
        input_string.append(f"iter {iterations}\n")
        input_string.append("vpar\n")
        input_string.append(f"n {n_crit}\n\n")

        # Setting polar name
        input_string.append("pacc\n")
        input_string.append(f"{polar_name}\n\n")

        # Setting alpha values
        input_string.append("aseq\n")
        input_string.append(f"{alphas[0]}\n")
        input_string.append(f"{alphas[1]}\n")
        input_string.append(f"{alphas[2]}\n")
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
        if os.path.exists(file_path):
            os.remove(file_path)
        else:
            logging.debug(f"File {file_path} not found")

    @staticmethod
    def plot_airfoil(file_path, separate_curves=False):
        """
        Plots airfoil using matplotlib
        :param file_path: path of the airfoil dat file
        :param separate_curves:
        :return:
        """
        logging.info("Plotting airfoil")
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

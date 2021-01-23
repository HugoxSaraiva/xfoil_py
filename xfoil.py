import concurrent.futures as future
import itertools
import logging
import numpy as np
import os
import re
import subprocess as sp
import sys
from matplotlib import pyplot as plt
from scipy.special import comb
from threading import Timer
from utils.user_options import UserOptions
from utils.utils import add_prefix_suffix, random_string, path_leaf
# TODO: use BezierSegment from matplotlib


class XFoil:
    def __init__(
            self,
            name,
            mach,
            reynolds,
            alpha_min,
            alpha_max,
            alpha_step,
            save_polar_name=None,
            executable_path="xfoil"
    ):
        logging.info("Initializing XFoil class")
        # Attributes to use with xfoil
        self.name = name

        if save_polar_name:
            self.save_polar_name = add_prefix_suffix(save_polar_name, suffix=".txt")
        else:
            self.save_polar_name = add_prefix_suffix(random_string(), prefix="tmp_", suffix=".txt")

        self.mach = mach
        self.reynolds = reynolds
        self.alpha_min = alpha_min
        self.alpha_max = alpha_max
        self.alpha_step = alpha_step
        self.results = None

        # Parameters not "directly" available to users
        self.panels = 300
        self.iter = 100
        self.n_crit = 9
        self.process_timeout = 30

        # Attributes to use in XFoil class
        self.executable = executable_path
        self.process = None
        self.stdout = None
        self.err = None
        self.disable_graphics = True

        debug_msg = "XFoil class initialized with properties: name: {} mach: {} reynolds: {} alphas: {} {} {}"
        logging.debug(debug_msg.format(self.name, mach, reynolds, alpha_min, alpha_max, alpha_step))

    def run(self):
        """
        Runs xfoil with given filename, mach, reynolds, alpha_min, alpha_max and alpha_step, used to initialize class
        :return: None
        """
        logging.info("XFoil class run() method called")

        # Check if there is an old txt file and remove it:
        self._file_cleanup()

        logging.info("Opening subprocess")
        # Open subprocess to run xfoil
        try:
            self.process = sp.Popen(
                self.executable,
                stdin=sp.PIPE,
                stdout=sp.PIPE,
                stderr=sp.PIPE
            )
        except FileNotFoundError:
            raise ExecutableNotFoundError(f"Executable {self.executable} not found")

        logging.info("Communicating input string to subprocess")
        # Kills process if it takes too long
        timer = Timer(self.process_timeout, self.__kill_process)
        try:
            timer.start()
            stdout, err = self.process.communicate("".join(self._input_string).encode())

            # Store stdout and err for debugging
            self.stdout = stdout.decode()
            self.err = err.decode() if err.decode() else None

            logging.info("Proceeding to read polar file")
            # Reading polar text file to get results
            try:
                self.results = self.read_polar(self.save_polar_name)
                if self.save_polar_name.startswith("tmp_"):
                    self._file_cleanup()
                else:
                    logging.info(f"Keeping polar file {self.save_polar_name} on disk")
            except FileNotFoundError:
                logging.warning("An error occurred while trying to read polar")
                logging.debug(f"Tried to read polar {self.save_polar_name} and failed")
                self.results = None
        finally:
            logging.info("XFoil class run() method ended")
            timer.cancel()

    def plot_polar(self, x_column_name, y_column_name, save_plot_name=None, plot_title=None):
        logging.info("Plotting polar")
        if not self.results:
            raise TypeError("XFoil result is None")
        plt.plot(x_column_name, y_column_name, data=self.results)
        plt.xlabel(x_column_name)
        plt.ylabel(y_column_name)
        plt.title(plot_title)

        # Stop blocking code execution when debugging
        if not self.debug:
            plt.show()
        if save_plot_name:
            plt.savefig(save_plot_name)

    def _file_cleanup(self):
        logging.debug(f"Deleting polar file {self.save_polar_name}")
        if os.path.exists(self.save_polar_name):
            os.remove(self.save_polar_name)

    @property
    def _is_naca(self):
        return self.__valid_naca(self.name)

    @property
    def _input_string(self):
        logging.info("Creating input string")
        input_string = []
        # Check whether to use naca from xfoil or load a dat file
        if self._is_naca:
            input_string.append(f"naca {self.name}\n")
        else:
            input_string.append(f"load {add_prefix_suffix(self.name, suffix='.dat')}\n")
            input_string.append(f"{self.name}\n")

        # Disabling airfoil plotter from appearing
        if self.disable_graphics:
            input_string.append("plop\n")
            input_string.append("g 0\n\n")

        # Setting number of panels
        input_string.append("ppar\n")
        input_string.append(f"n {self.panels}\n\n\n")

        # Setting viscous calculation parameters
        input_string.append("oper\n")
        input_string.append(f"visc {self.reynolds}\n")
        input_string.append(f"M {self.mach}\n")
        input_string.append(f"iter {self.iter}\n")
        input_string.append("vpar\n")
        input_string.append(f"n {self.n_crit}\n\n")

        # Setting polar name
        input_string.append("pacc\n")
        input_string.append(f"{self.save_polar_name}\n\n")

        # Setting alpha values
        input_string.append("aseq\n")
        input_string.append(f"{self.alpha_min}\n")
        input_string.append(f"{self.alpha_max}\n")
        input_string.append(f"{self.alpha_step}\n")
        input_string.append("\nquit\n")

        logging.debug(f"Input string is : {input_string}")
        return input_string

    def __read_stdout(self):
        logging.info("Reading stdout")
        exp = "((?<=x[/]c\s=\s{2})\d\.\d*|\d+(?=\s{3}rms)|" \
              "(?<=(?:Cm|CL|CD)\s=\s)[\s-]?\d*\.\d*|" \
              "(?<=a\s=\s)[\s-]?\d*\.\d*|" \
              "(?<=CDp\s=\s)[\s-]?\d*\.\d*)"
        regex = re.compile(exp)

        data = regex.findall(self.stdout)
        a = []
        cl = []
        cd = []
        cdp = []
        cm = []
        xtr_top = []
        xtr_bottom = []

        # Pick data from only the last iteration for each angle
        i = 0
        while i < len(data):
            iteration = float(data[i + 2])
            if i + 10 > len(data) or iteration > float(data[i + 10]):
                xtr_top.append(float(data[i]))
                xtr_bottom.append(float(data[i + 1]))
                a.append(float(data[i + 3]))
                cl.append(float(data[i + 4]))
                cm.append(float(data[i + 5]))
                cd.append(float(data[i + 6]))
                cdp.append(float(data[i + 7]))
            i = i + 8
        return {'a': np.array(a), 'cl': np.array(cl), 'cd': np.array(cd), 'cdp': np.array(cdp), 'cm': np.array(cm),
                'xtr_top': np.array(xtr_top), 'xtr_bottom': np.array(xtr_bottom)}

    def __kill_process(self):
        self.process.kill()
        logging.warning("Process killed due to timeout")

    @staticmethod
    def __valid_naca(string):
        """
        Gets an input string and returns true if it is a 4 or 5 digit naca number and returns false if it isn't
        :param string: string
        :return: boolean
        """
        return string.isdigit() and (len(string) in [4, 5])

    @staticmethod
    def __bezier_curve_2d(control_points):
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

        regex = re.compile('(?:\s*([+-]?\d*.\d*))')
        with open(file_path) as f:
            lines = f.readlines()

            a = []
            cl = []
            cd = []
            cdp = []
            cm = []
            xtr_top = []
            xtr_bottom = []

            for line in lines[12:]:
                line_data = regex.findall(line)
                if line_data:
                    try:
                        a.append(float(line_data[0]))
                        cl.append(float(line_data[1]))
                        cd.append(float(line_data[2]))
                        cdp.append(float(line_data[3]))
                        cm.append(float(line_data[4]))
                        xtr_top.append(float(line_data[5]))
                        xtr_bottom.append(float(line_data[6]))
                    except IndexError:
                        raise InvalidFileContentsError(f"No valid polar found in file: '{file_path}'")

            polar_contents = {
                'a': np.array(a),
                'cl': np.array(cl),
                'cd': np.array(cd),
                'cdp': np.array(cdp),
                'cm': np.array(cm),
                'xtr_top': np.array(xtr_top),
                'xtr_bottom': np.array(xtr_bottom)
            }
            if any([len(values) == 0 for _, values in polar_contents.items()]):
                raise InvalidFileContentsError(f"No valid polar found in file: '{file_path}'")
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
        upper_curve = XFoil.__bezier_curve_2d(upper_control_points)
        lower_curve = XFoil.__bezier_curve_2d(lower_control_points)

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


def __run_xfoil(name, mach, reynolds, save_polar_name, args, plot=False):
    xfoil = XFoil(
        name,
        mach,
        reynolds,
        args.alphas[0],
        args.alphas[1],
        args.alphas[2],
        save_polar_name,
        args.executable_path
    )
    xfoil.run()
    if plot:
        xfoil.debug = args.debug
        xfoil.plot_polar(args.plot[1], args.plot[0], args.save_plot_name, args.save_plot_title)
    return xfoil.results


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

    # Support running multiple airfoils with mach/reynolds configurations in parallel
    if len(args.name) > 1 or len(args.mach) > 1 or len(args.reynolds) > 1:
        fill_value = args.mach[-1] if len(args.mach) <= len(args.reynolds) else args.reynolds[-1]
        mach_reynolds_iterator = itertools.zip_longest(args.mach, args.reynolds, fillvalue=fill_value)
        args_iterator = itertools.product(args.name, mach_reynolds_iterator)

        save_polar_name = args.save_polar_name.replace(".txt", "") + "-N-{}-M-{}-R-{}"

        with future.ThreadPoolExecutor(max_workers=args.max_threads) as executor:
            # Start the load operations and mark each future with its URL
            futures = [
                executor.submit(__run_xfoil,
                                name,
                                mach,
                                reynolds,
                                save_polar_name.format(path_leaf(name), str(mach).replace('.', ''), reynolds),
                                args
                                ) for name, (mach, reynolds) in args_iterator
            ]
            executor.shutdown(wait=True)
            return [thread.result() for thread in futures]
    else:
        # Running xfoil only once, save_polar_name can be simpler
        return __run_xfoil(args.name[0], args.mach[0], args.reynolds[0], args.save_polar_name, args, True)


if __name__ == "__main__":
    main(sys.argv[1:])

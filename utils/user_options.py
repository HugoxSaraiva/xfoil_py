import argparse


class UserOptions:
    def __init__(self):
        self.__parsed_arguments = None

        # Define parser
        parser = argparse.ArgumentParser(description='xfoil wrapper for python.')
        parser.add_argument('-n', '--name', type=str, dest="name", required=True, nargs='+',
                            help='Path to .dat file or number of NACA airfoil')
        parser.add_argument('-m', '--mach', type=float, dest="mach", required=True, nargs='+',
                            help="Mach used to run xfoil")
        parser.add_argument('-r', '--reynolds', type=int, dest="reynolds", required=True, nargs='+',
                            help="Reynolds used to run xfoil.")
        parser.add_argument('-a', '--alpha', type=float, dest="alphas", nargs='+', default=[0, 15, 0.5],
                            help='alpha_min, alpha_max, alpha_step used to run xfoil, default is 0, 15, 0.5')
        parser.add_argument('-s', '--save-polar', type=str, dest="save_polar_name",
                            help="Save name or path of xfoil polar file. Names with 'tmp_' are deleted automatically."
                                 "\nSave name will be 'save_name-N-name-M-mach-R-reynolds-A-alphamin-alphamax-step' "
                                 "if xfoil is running multiple runs, or save_name otherwise")
        parser.add_argument("-x", "--executable", type=str, dest="executable_path", default=None,
                            help="Path to executable. Default behaviour is to use xfoil on /runs directory")
        parser.add_argument("-t", "--threads", type=int, dest="max_threads", default=4,
                            help="Max threads to run xfoil in parallel. Default value is 4.")
        parser.add_argument('-p', '--plot', type=str, dest="plot", nargs=2,
                            choices=['a', 'cl', 'cd', 'cdp', 'cm', 'xtr_top', 'xtr_bottom'],
                            help="Plots variables from xfoil result. 'cl a' plots Cl x alpha polar")
        parser.add_argument("--p-n", "--plot-name", type=str, dest="save_plot_name",
                            help="Save name of polar plot file")
        parser.add_argument("--p-t", "--plot-title", type=str, dest="save_plot_title",
                            help="Title of plot")
        parser.add_argument("-d", "--debug", action='store_true', help='Print debug messages')

        # Doesn't require xfoil to run results
        parser.add_argument('--show', dest="show", action='store_true',
                            help="Plots airfoil profile from .dat. Mach and Reynolds are needed but not used")

        # declare the internal argparse parser
        self.__parser = parser

    def parse_arguments(self, args):
        self.__parser.parse_args(args, namespace=self)

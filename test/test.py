import unittest
import pandas as pd
import numpy as np
import os
import itertools
import xfoil


class XFoilTestCase(unittest.TestCase):
    fast_test = False

    # Test if XFoil reads dat properly
    def test_read_dat(self):
        names = {
            "naca0012",
            "naca0012-no-name"
        }
        for name in names:
            # Check if first line is the airfoil name:
            df1 = pd.read_csv(f"data/{name}.dat", header=None, sep=" ", nrows=1, skipinitialspace=True)
            have_name_line = type(df1[0][0]) == str
            skip_row = 1 if have_name_line else 0

            # Read dat
            dat_dataframe = pd.read_csv(
                f"data/{name}.dat",
                sep=" ",
                names=['x', 'y'],
                skiprows=skip_row,
                skipinitialspace=True
            )
            dat = pd.DataFrame.from_dict(xfoil.XFoil.read_dat(f"data/{name}.dat"))
            self.assertTrue(np.isclose(dat.values, dat_dataframe.values).all())

    # Test if XFoil creates dat-file properly
    def test_create_dat(self):
        upper_control_points = np.load("data/upper_control_points.pickle", allow_pickle=True)
        lower_control_points = np.load("data/lower_control_points.pickle", allow_pickle=True)
        xfoil.XFoil.create_airfoil(
            "test",
            upper_control_points,
            lower_control_points
        )

        created_dat_dataframe = pd.read_csv(
            "test.dat",
            sep=" ",
            names=['x', 'y'],
            skiprows=0,
            skipinitialspace=True
        )
        expected_dat_dataframe = pd.read_csv(
            "data/NATAFOIL.dat",
            sep=" ",
            names=['x', 'y'],
            skiprows=0,
            skipinitialspace=True
        )

        if os.path.exists("test.dat"):
            os.remove("test.dat")
        self.assertTrue(np.isclose(created_dat_dataframe.values, expected_dat_dataframe.values).all())

    # Test if xfoil.exe can run
    def test_run(self):
        # Check if xfoil exists on path
        # Test if right error is raised when the executable is not found.
        args = "-n 0012 -m 0.5 -r 31000000 -a -5 15 0.5 -x does-not-exist".split()
        self.assertRaises(xfoil.ExecutableNotFoundError, xfoil.main, args)

    # Test if XFoil class is returning the correct result
    @unittest.skipIf(fast_test, "Skip testing data for a faster test")
    def test_results(self):
        args_test_1 = "-n 0012 4412 4508 -m 0.5 -r 31000000 -a -5 15 0.5 -s tmp_args_run -d".split()
        args_test_2 = "-n 0012 4412 4508 -m 0.2 -r 18000000 -a 0 15 1 -s tmp_args_run -d".split()
        nacas = [
            "0012",
            "4412",
            "4508"
        ]
        test_cases = [
            [0.5, 31000000, -5, 15, 0.5],
            [0.2, 18000000, 0, 15, 1]
        ]
        results = xfoil.main(args_test_1)
        results.extend(xfoil.main(args_test_2))
        idx = [0, 3, 1, 4, 2, 5]
        # Checking if files created have correct results
        for i, (naca, test_case) in enumerate(itertools.product(nacas, test_cases), 1):
            xfoil_fortran = pd.read_csv(
                f"data/{naca}-{i}.txt",
                sep=" ",
                skipinitialspace=True,
                skiprows=[x for x in range(12) if x != 10]
            )
            xfoil_python = pd.DataFrame.from_dict(results[idx[i-1]])
            xfoil_python.columns = xfoil_fortran.columns

            # Using np.isclose() to check if values in array are close ignoring floating point errors
            self.assertTrue(np.isclose(xfoil_python.values, xfoil_fortran.values).all())

    @unittest.skipIf(fast_test, "Skip testing args input for a faster test")
    def test_args(self):
        # Test use case
        args = "-n 0012 -m 0.5 -r 31000000 -a -5 15 0.5 -s args_run -p cl a --p-n saved_plot --p-t 'Test' -d".split()
        xfoil.main(args)
        xfoil_fortran = pd.read_csv(
            f"data/0012-1.txt",
            sep=" ",
            skipinitialspace=True,
            skiprows=[x for x in range(12) if x != 10]
        )
        xfoil_args = pd.read_csv(
            f"args_run.txt",
            sep=" ",
            skipinitialspace=True,
            skiprows=[x for x in range(12) if x != 10]
        )
        if os.path.exists("args_run.txt"):
            os.remove("args_run.txt")
        self.assertTrue(np.isclose(xfoil_args.values, xfoil_fortran.values).all())

        # Test if plot is saved
        self.assertTrue(os.path.exists("saved_plot.png"))
        if os.path.exists("saved_plot.png"):
            os.remove("saved_plot.png")

    def test_dat_run(self):
        xfoil_object = xfoil.XFoil("data/NATAFOIL.dat", 0.5, 31000000, -5, 10, 0.2)
        xfoil_object.run()

        xfoil_fortran = pd.read_csv(
            f"data/natafoil.txt",
            sep=" ",
            skipinitialspace=True,
            skiprows=[x for x in range(12) if x != 10]
        )
        xfoil_python = pd.DataFrame.from_dict(xfoil_object.results)
        xfoil_python.columns = xfoil_fortran.columns

        # Using np.isclose() to check if values in array are close ignoring floating point errors
        self.assertTrue(np.isclose(xfoil_python.values, xfoil_fortran.values).all())


if __name__ == '__main__':
    unittest.main()

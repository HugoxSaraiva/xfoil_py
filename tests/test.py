import unittest
import pandas as pd
import numpy as np
import os
import itertools
from xfoil_py import xfoil


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
        args = "-n 0012 4412 4508 -m 0.5 0.2 -r 31000000 18000000 -a -5 15 0.5 0 15 1 -t 6 -d".split()
        nacas = [
            "0012",
            "4412",
            "4508"
        ]
        test_cases = [
            [0.5, 31000000, -5, 15, 0.5],
            [0.2, 18000000, 0, 15, 1]
        ]
        results = xfoil.main(args)
        # Checking if files created have correct results
        for i, (naca, test_case) in enumerate(itertools.product(nacas, test_cases)):
            xfoil_python = pd.DataFrame.from_dict(results[i]['result'])

            xfoil_fortran_sp = pd.read_csv(
                f"data/{naca}-{i+1}.txt",
                sep=" ",
                skipinitialspace=True,
                skiprows=[x for x in range(12) if x != 10]
            )
            xfoil_fortran_dp = pd.read_csv(
                f"data/{naca}-{i+1}-DP.txt",
                sep=" ",
                skipinitialspace=True,
                skiprows=[x for x in range(12) if x != 10]
            )
            if xfoil_python.shape == xfoil_fortran_dp.shape:
                print("Double Precision detected")
                xfoil_fortran = xfoil_fortran_dp
            else:
                xfoil_fortran = xfoil_fortran_sp
            # Using np.isclose() to check if values in array are close ignoring floating point errors
            self.assertTrue(np.isclose(xfoil_python.values, xfoil_fortran.values).all())

    @unittest.skipIf(fast_test, "Skip testing args input for a faster test")
    def test_args(self):
        # Test use case
        args = "-n 0012 -m 0.5 -r 31000000 -a -5 15 0.5 -s args_run -d".split()
        xfoil.main(args)
        xfoil_args = pd.read_csv(
            "0_args_run.txt",
            sep=" ",
            skipinitialspace=True,
            skiprows=[x for x in range(12) if x != 10]
        )

        # Verifying if xfoil executable is in double precision (dp) or not:
        xfoil_fortran_sp = pd.read_csv(
            f"data/0012-1.txt",
            sep=" ",
            skipinitialspace=True,
            skiprows=[x for x in range(12) if x != 10]
        )
        xfoil_fortran_dp = pd.read_csv(
            f"data/0012-1-DP.txt",
            sep=" ",
            skipinitialspace=True,
            skiprows=[x for x in range(12) if x != 10]
        )
        if len(xfoil_args) == len(xfoil_fortran_dp):
            print("Double Precision detected")
            xfoil_fortran = xfoil_fortran_dp
        else:
            xfoil_fortran = xfoil_fortran_sp

        if os.path.exists("0_args_run.txt"):
            os.remove("0_args_run.txt")
        self.assertTrue(np.isclose(xfoil_args.values, xfoil_fortran.values).all())

    def test_dat_run(self):
        xfoil_object = xfoil.XFoil("data/NATAFOIL.dat", 0.5, 31000000, [-5, 10, 0.2])
        xfoil_object.run()

        xfoil_fortran_sp = pd.read_csv(
            f"data/natafoil.txt",
            sep=" ",
            skipinitialspace=True,
            skiprows=[x for x in range(12) if x != 10]
        )
        xfoil_fortran_dp = pd.read_csv(
            f"data/natafoil-DP.txt",
            sep=" ",
            skipinitialspace=True,
            skiprows=[x for x in range(12) if x != 10]
        )
        xfoil_python = pd.DataFrame.from_dict(xfoil_object.results[0]['result'])

        if len(xfoil_python) == len(xfoil_fortran_dp):
            print("Double Precision detected")
            xfoil_fortran = xfoil_fortran_dp
        else:
            xfoil_fortran = xfoil_fortran_sp

        xfoil_python.columns = xfoil_fortran.columns

        # Using np.isclose() to check if values in array are close ignoring floating point errors
        self.assertTrue(np.isclose(xfoil_python.values, xfoil_fortran.values).all())


if __name__ == '__main__':
    unittest.main()

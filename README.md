# xfoil_py
> xfoil_py is a python project that makes the use of the xfoil more accessible in Python.

## Table of contents
* [General info](#general-info)
* [Screenshots](#screenshots)
* [Setup](#setup)
* [Features](#features)
* [Status](#status)
* [Inspiration](#inspiration)
* [Contact](#contact)

## General info
The xfoil executable (coded in FORTRAN) is known by its usefulness and accuracy in calculating aerodynamic coefficients,
such as the lift coefficient and the drag coefficient. xfoil_py is intended to further improve the usability of xfoil by 
enabling parallel computing for multiple airfoils and test cases, and by reading the results automatically.
xfoil_py can be used as a script or imported as a module.

## Screenshots
None at the moment

## Setup
###Linux:
Users must certify that they have xfoil running correctly before using this project.

For this project to be able to run, packages in requirements.txt must be installed in your Python environment.
This can be done by opening a terminal in the project root and typing:

    pip install -r requirements.txt

xfoil_py assumes by default that the xfoil binary is located in the "runs" folder. A symbolic link of the previously 
installed xfoil binary can be created in the runs folder to enable xfoil_py to use it by default. The user can
also specify the argument -x "PATH_TO_BINARY" when running xfoil.py as a script or specify "executable_path" when 
instantiating an XFoil class.

###Windows:
Windows' users don't need to have xfoil.exe previously installed since the executable comes in the "runs" folder.
All that is needed is to have the Python packages on requirements.txt installed in your Python environment.
This can be done by opening a terminal in the project root and typing:

    pip install -r requirements.txt

## Code Examples
Examples of usage:

###xfoil.py as a script:
Runs xfoil on NACA 0012 airfoil with parameters: 

Mach = 0.3, Reynolds = 9000000, alpha_min = -5, alpha_max = 15, alpha_step = 0.5.

Also saves the polar as the file 'polar.txt'

    python -m xfoil.py -n 0012 -m 0.3 -r 9000000 -a -5 15 0.5 -s polar

###xfoil.py as a script with multiple testcases:
Runs xfoil on NACA 0012, 4412 and 4508 twice with parameters:

|   |Mach|Reynolds|alpha_min|alpha_max|alpha_step|
|---|---|---|---|---|---|
|Test case 1|0.5|31000000|-5|15|0.5|
|Test case 2|0.2|18000000|0|15|1|

Also saves the polar files with the prefix 'polar'

        python -m xfoil.py -n 0012 4412 4508 -m 0.5 0.2 -r 31000000 18000000 -a -5 15 0.5 0 15 1 -s polar

This will generate polar files :

polar-N-0012-M-0_2-R-18000000-A-0_0-15_0-1_0.txt\
polar-N-0012-M-0_5-R-31000000-A--5_0-15_0-0_5.txt\
polar-N-4412-M-0_2-R-18000000-A-0_0-15_0-1_0.txt\
polar-N-4412-M-0_5-R-31000000-A--5_0-15_0-0_5.txt\
polar-N-4508-M-0_2-R-18000000-A-0_0-15_0-1_0.txt\
polar-N-4508-M-0_5-R-31000000-A--5_0-15_0-0_5.txt

###XFoil class:
Loads a dat file in "data" folder and runs it with parameters:

Mach = 0.5, Reynolds = 31000000, alpha_min = -5, alpha_max = 10, 
alpha_step = 0.2.

    from xfoil import XFoil
    xfoil_object = XFoil("data/NATAFOIL.dat", 0.5, 31000000, -5, 10, 0.2)
    xfoil_object.run()

    # Getting results as a dict:
    results = xfoil_object.results
    print(results)

## Features
* Support for parallel computing
* Processes polar file content automatically
* Identifies if airfoil name is a dat file, or a 4-5 digit NACA number
* Reads dat files and returns as a dict with 'x' and 'y' coordinates
* Plots airfoil from dat file

To-do list:
* Add support of parallel computing from inside the XFoil class with improved return of results
* Add support for other use cases of xfoil  
* Add an automatic setup of xfoil_py for Windows and Linux.
* Improve plotting options
* Compile and use fortran files directly instead of reading output file from disk to improve speed.

## Status
Project is: _in progress_

## Inspiration
Project inspired by https://github.com/daniel-de-vries/xfoil-python.

## Contact
Created by [@HugoxSaraiva](https://twitter.com/HugoxSaraiva) - feel free to contact me!
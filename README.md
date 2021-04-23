# xfoil_py
> xfoil_py is a python package that makes the use of the xfoil more accessible in Python.

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
enabling parallel computing of multiple airfoils and test cases, and by reading the results automatically.
xfoil_py can be used as a script or imported as a package.

## Screenshots
None at the moment

## Setup
###### Linux:
**(In development)\
xfoil_py can only run on linux as a script at the moment.**
\
\
Users must certify that they have xfoil running correctly before using this project.

For this project to be able to run, packages in requirements.txt must be installed in your Python environment.
This can be done by opening a terminal in the folder xfoil_py inside the project root and typing:

    pip install -r requires.txt

xfoil_py assumes by default that the xfoil binary is located in the "runs" folder. A symbolic link of the previously 
installed xfoil binary can be created in the runs folder for it to be used by default. The user can
also specify the argument -x "PATH_TO_BINARY" when running xfoil.py as a script or with the keyword argument 
executable_path="PATH_TO_BINARY" when instantiating an XFoil class.

###### Windows:
Windows' users don't need to have xfoil.exe previously installed since the executable comes in the "runs" folder.
All that is needed is to run the setup file in order to have all required packages installed in your Python environment.
This can be done by opening a terminal in the project root and typing:

    pip install -r requires.txt

## Code Examples
###### xfoil.py as a script:
Runs xfoil on NACA 0012 airfoil with parameters: 

Mach = 0.3, Reynolds = 9000000, alpha_min = -5, alpha_max = 15, alpha_step = 0.5.

Also saves the polar as the file 'polar.txt'

    python -m xfoil.py -n 0012 -m 0.3 -r 9000000 -a -5 15 0.5 -s polar

###### xfoil.py as a script with multiple testcases:
Runs xfoil on NACA 0012, 4412 and 4508 twice with parameters:

|   |Mach|Reynolds|alpha_min|alpha_max|alpha_step|
|---|---|---|---|---|---|
|Test case 1|0.5|31000000|-5|15|0.5|
|Test case 2|0.2|18000000|0|15|1|

Also saves the polar files with the suffix 'polar.txt'

        python -m xfoil.py -n 0012 4412 4508 -m 0.5 0.2 -r 31000000 18000000 -a -5 15 0.5 0 15 1 -s polar

This will generate polar files :

0_polar.txt corresponding to run on NACA 0012 with test case 1\
1_polar.txt corresponding to run on NACA 0012 with test case 2\
2_polar.txt corresponding to run on NACA 4412 with test case 1\
3_polar.txt corresponding to run on NACA 4412 with test case 2\
4_polar.txt corresponding to run on NACA 4508 with test case 1\
5_polar.txt corresponding to run on NACA 4508 with test case 2

###### XFoil class:
Loads a dat file in "data" folder and runs it with parameters:

Mach = 0.5, Reynolds = 31000000, alpha_min = -5, alpha_max = 10, 
alpha_step = 0.2.

    from xfoil_py import XFoil
    xfoil_object = XFoil("data/NATAFOIL.dat", 0.5, 31000000, [-5, 10, 0.2])
    xfoil_object.run()

    # Getting results as a dict:
    results = xfoil_object.results[0]['result']
    print(results)

## Features
* Support for parallel computing
* Processes polar file content automatically
* Identifies if airfoil name is a dat file, or a 4-5 digit NACA number
* Reads dat files and returns as a dict with 'x' and 'y' coordinates
* Plots airfoil from dat file

To-do list:
* Add support for other use cases of Xfoil  
* Compile and use fortran files directly instead of reading output file from disk to improve speed.
* Add an automatic setup of xfoil_py (including compiling Xfoil) for Windows and Linux.
* Create plotting routines to help users plotting results

## Status
Project is: _in progress_

## Inspiration
Project inspired by https://github.com/daniel-de-vries/xfoil-python.

## Contact
Created by [@HugoxSaraiva](https://twitter.com/HugoxSaraiva) - feel free to contact me!
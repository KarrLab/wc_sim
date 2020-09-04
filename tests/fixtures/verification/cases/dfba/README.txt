This directory contains original SBML Test Suite FluxBalanceSteadyState test cases that have been modified into dFBA test cases for use by wc_sim/tests/testing/test_verify.py to test wc_sim/testing/verify.py. 


Modifications
-------------


### Settings files

The original `NNNNN-settings.txt` files were renamed as `NNNNN-settings_static.txt`.

New `NNNNN-settings.txt` files were created to follow the format of `TimeCourse` tests, and the parameter values were set as follows:

* `start` was set to 0 

* `duration` was set to 1, 10, or 100 to ensure integer values during species population update at the end of the duration

* `steps` was set to 1

* `variables` contains a list of boundary species IDs whose values are tabulated in the CSV file

* `absolute` was set to the original values in the FluxBalanceSteadyState test cases

* `relative` was set to the original values in the FluxBalanceSteadyState test cases

* `amount` contains the same list as `variables`

*` `concentration` was left blank


### Results files

The original `NNNNN-results.csv` files were renamed as `NNNNN-results_static.csv`

New `NNNNN-results.csv` files were created to follow the format of `TimeCourse` tests, showing the initial species populations at time 0 and the final species populations at the end of the simulation duration. All species were set to an initial amount of 1000. The final populations were calculated as the multiplication of the reaction flux solutions in the original test cases, simulation duration, and the stoichiometric coefficients.


Creating test models from the test cases
--------------------------------------------

The SBML test files are read and converted into WC-Lang models at run time during verification. The species, reactions, flux bounds and objective functions are converted accordingly. The initial amounts of all species are set to 1000. To enable accumulation of boundary species, exchange reactions are created for these species as dfba-objective-reactions and added to the dfba-objective with a linear coefficient of 0. 


 
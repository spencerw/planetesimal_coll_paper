This folder contains the data for all five simulations presented in the paper. A shell script which downloads the data
will be available once the data are made available via the UW Library.

Each directory contains the following:

- A series of simulation snapshots
- The log files for the recorded collisions
- A '.param' file, which contains the simulation parameters that were passed to ChaNGa
- A '.par' file, which contains the parameters used to generate the initial conditions
- A '.out' file, which contains output information from the IC generator

The initial condition file generator is available at:
https://github.com/spencerw/makeSecICs

Note: Each simulation was run for 2000 years with collision detection turned off, in order to allow the resonances
to fully develop. Collisions were then recorded for another 3000 years. The simulation snapshots provided here begin
at the T=2000 and end at T=5000 yr.

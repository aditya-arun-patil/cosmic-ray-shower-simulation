Cosmic Ray Air Shower Simulation

A simple 1D Monte Carlo simulation in Python for modeling Cosmic Ray Extensive Air Showers (EAS) and analyzing the results.
This project is split into two main parts:
cosmic_ray_shower.py: The simulation engine. It models a primary proton (or other hadron) striking the atmosphere and simulates the resulting particle cascade.
analyze_shower_data.py: The analysis script. It reads the output data from the simulation and generates plots to visualize the physical properties of the showers.

Features
Simulation (cosmic_ray_shower.py)
Models a 1D exponential atmosphere.
Simulates hadronic interactions (pions, kaons, nucleons) using parameterized models.
Simulates electromagnetic cascades (pair production, bremsstrahlung) driven by $\pi^0$ decays.
Models charged pion decay ($\pi^\pm \to \mu^\pm$) and muon propagation.
Tracks continuous energy loss from ionization for all charged particles.
Saves all final-state particles from many showers into a single .csv file.
Analysis (analyze_shower_data.py)
Reads the simulation .csv file using Pandas.
Generates plots for key physical observables:
Average Longitudinal Profile (Particle count vs. Atmospheric Depth)
Shower Maximum (Xmax) Distribution
Ground Particle Count Distribution
Ground Particle Energy Spectrum (separating EM and Muonic components)
Muon-to-EM Ratio vs. Depth

How to Run
1. Requirements
You will need Python 3 and a few common scientific libraries. You can install them using pip:


pip install numpy pandas matplotlib scipy


2. Run the Simulation
Run the simulation script from your terminal. It will prompt you to enter the primary energy (in GeV) and the number of showers to simulate.


python cosmic_ray_shower.py


This will run for some time and produce a .csv output file, for example: shower_data_100GeV_1000runs.csv.

3. Run the Analysis

Once the simulation is complete, run the analysis script. You must provide the name of the .csv file as a command-line argument.


python analyze_shower_data.py   shower_data_100GeV_1000runs.csv


This will read the data and save 5 plots as .png files in the same directory (e.g., avg_longitudinal_profile.png, xmax_distribution.png, etc.).


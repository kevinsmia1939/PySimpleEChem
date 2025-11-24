# PySimpleEChem

Graphical user interface for plotting and analysing the voltammograms with simple-to-use sliders.<br />
Written in pure Python.<br />

PySimpleEChem is written in PyQt5 to replace PySimpleCV, which uses PySimpleGUI because PySimpleGUI has changed to a proprietary license.<br />
License: GPLv3 <br />

# Features

PySimpleEChem is very much in pre-alpha and is not ready for everyday usage.<br />
Feel free to make a bug report for new features <br />

**Cyclic voltammetry**
* Select and plot multiple CV at the same time.<br />
* Currently supports VersaStudio (.par), Correware(.cor), .csv, and .txt. Please send some other format for more file format support. For .csv and .txt, the first column must be voltage and the second column is current.<br />

# Future plans

* Nicholson method to calculate peak current when the base line cannot be determine.<br />
* Calculate diffusion coefficient and rate of reaction from Randles-Sevcik equation.<br />
* Plot peak current vs. sqare root of scan rate for diffustion coefficient.<br />
* Plot peak current vs. peak separation for rate of reaction.<br />
* Detect peak with maximum, minimum, or 2nd derivatives.<br />
* Export results and save file.<br />

**Cyclic voltammetry ECSA **

* Calculate electrochemical active surface area (ECSA) with selected area under the CV. <br />

**Rotating Disk Electrode**

*Calculate diffusion coefficient and kinetic current from Levich equation. <br />

# PyLimitingCurrent.py

Graphical user interface for measuring limiting current from linear sweep voltammogram (LSV) with simple-to-use slider. Limiting current was measured to determine the mass transport coefficient.
The method to determine limiting current is by plotting LSV as 1/I on x-axis and E/I on y-axis. The puurpose of plotting LSV like this is to make determining limiting current more reproducible.

See this for more detail
Ponce-de-León, C., Low, C.T.J., Kear, G. et al. Strategies for the determination of the convective-diffusion limiting current from steady state linear sweep voltammetry. J Appl Electrochem 37, 1261–1270 (2007). https://doi.org/10.1007/s10800-007-9392-3

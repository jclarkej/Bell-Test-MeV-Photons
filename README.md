# Bell Test for MeV Photons via POVM-based Compton Polarimetry

This repository contains the Python code written by Jack Clarke (UCL) for the manuscript: **"Bell Test for MeV Photons via POVM-based Compton Polarimetry"**.

## Description

We study a POVM-based analysis of photon polarimetry via sequential Compton scattering. Scattering trajectories are found where the Compton analyzing power $`\beta`$ (which determines the polarization measurement sharpness) is enhanced. 

This POVM-analysis can be used to describe a Bell test where Alice and Bob each perform unsharp polarization measurements on separate MeV photons originating from electron-positron annihilation. For $`N`\geq2$ sequential scattering events per side, $`\beta`$ can be enhanced beyond the critical threshold and violations of the CHSH inequality may be realized.

The script *Bell_test_MeV.py* reproduces the numerical results presented in the manuscript including: Fig.2 of the main text, Tables I and II of the End Matter, and Table I of the Supplemental Material. This code may also be used to confirm the monotonic trend of $`\beta\rightarrow1`$ and $`|S|\rightarrow 2\sqrt{2}`$ as $`N`$ increases, in which case the number of maximum iterations in the **`SLQP OPTIMIZATION`** code block should be increased.

Depending on the objective selected, the code will optimize: (i) fidelity, (ii) trace norm, or (iii) $`\beta`$ (fastest option). Each objective gives the same optimization, as shown in the Supplemental Material (the fidelity/trace norm is maximized/minimized when $`\beta`$ is maximized).

The code contains various functions, but by uncommenting various lines of the code in the **`EXECUTION`** code block (at the very end of the *Bell_test_MeV.py* file), users can directly find the outputs of the most relevant functions for the manuscript:

* Optimal polar scattering angles for $`N`$ sequential scattering events
* Energy decay over $`N`$ sequential scattering events
* Max value of the CHSH function for $`N`$ sequential scattering events
* Plot of the CHSH function versus phi up to arbitrary $`N`$
* Functions to save the figures and data files

The POVM-based analysis of sequential scattering presented here may be readily generalized to other QFT scattering processes. 

## Requirements

The code requires following packages:

* NumPy
* SciPy
* Matplotlib

## Citation

When using this code, please cite our manuscript (this citation is to be updated when the journal version is available):
[[Jack Clarke, Preslav Asenov, Jesse Smeets, Jia-Shian Wang, David B. Cassidy, and Alessio Serafini, *"Bell Test of Photons from Electron-Positron Annihilation via POVM-based Compton Polarimetry"*, arXiv:2604.25034 (2026)]](https://arxiv.org/abs/2604.25034).





# Vibrational Spectroscopy

## Theory
The Raman tensor $\alpha$ can be approximated as:<br><br>
$\alpha(\omega_0)\approx\dfrac{\partial \epsilon(\omega_0)}{\partial Q}=\dfrac{\epsilon_{q^+}(\omega_0)-\epsilon_{q^-}(\omega_0)}{2\Delta q},$<br><br>
with $\omega_0$ being the laser wavelength, $\epsilon$ the dielectric function and $Q$ a phonon eigenmode. The right hand side is a numerical implementation of the differential for ionic displacements in + and - direction of the phonon mode $Q$ by a distance of $\Delta u$. As momentum has to be conserved, only phonons near $\Gamma$ can contribute to Raman scattering (in a first approximation at least).<br><br>
The Stokes intensity can then be calculated as:<br><br>
$I\sim|\hat{e}_s\alpha\hat{e}_i|^2\dfrac{(\omega_0-\omega_p)^4}{\omega_p}(n+1),$<br><br>
with $n$ being the Bose-Einstein occupation number, $\omega_p$ the phonon frequency and $\hat{e}$ the polarization direction of the incident and scattered light respectively. Both resonant and non-resonant Raman spectra can be obtained in this formulation.
<br><br>
Infrared spectra can be calculated via the effective charges $Z_a$ of ion $a$ and the phononic eigenvectors $\hat{Q}$. The imaginary part of the ionic contribution to the dielectric function (i.e. phononic absorption) is calculated by:<br><br>
$\epsilon_{ij}(\omega_p)\sim\left(\sum\limits_{a,k}Z_{a,ik}\hat{Q}_{p,a}\right)\left(\sum\limits_{a,k}Z_{a,jk}\hat{Q}_{p,a}\right).$<br><br>
The corresponding real part can be calculated using Kramers-Kronig relations.
<br><br>


## Phonon modes and frequencies at Γ-point
In order to start the script you need the phononic eigenmodes at Γ. Usage of the $phonopy$ format is recommended, but VASP is supported as well. The following files are needed:
- FORCE_CONSTANTS
- phonopy.yaml
- qpoints.yaml

Phonopy writes the $phonopy.yaml$ file per default. FORCE_CONSTANTS can be calculated from FORCE_SETS or different DFT calculators by
```bash
phonopy --writefc --dim="x y z"
```
where $x,y,z$ are the dimensions of the supercell used, if finite-displacement method is chosen. The dynamical matrix (qpoints.yaml) at Γ can be obtained by
```bash
phonopy --readfc --writedm --qpoints=\"0 0 0\"
```

Further information is written in the documentation at https://phonopy.github.io/phonopy/.
<br>
If you have calculated the phonon modes in VASP, you just need the OUTCAR file containing the phononic eigenvectors and eigenfrequencies.
- OUTCAR

<br><br>

## Raman spectroscopy
- Prepare your structures by displacing the ions along the phonon eigenvector in plus and minus direction, for each phonon mode
```bash
python RamanPy -d
```
This creates folders for all considered modes and displacements (only one folder for each direction, i.e. two folder per mode). As a default, silent modes are ignored. Make sure to include the necessary files specified in the following in the parent folder where you run RamanPy.

- The electronic contribution to the dielectric function needs to be calculated for all created structurs. For VASP, a possible INCAR looks like this:
```bash
LOPTICS = .TRUE. # calculate the dielectric function as a sum over bands
NBANDS  = ...    # number of bands, check for convergence
EDIFF   = 1.e-8  # low value for accurate eigenvalues
ISMEAR  = 0      # Do not use -5 !
SIGMA   = 0.02   # depends on system, 0.02 (VASP default) should be fine
```
As for all optical calculations, check for k-point convergence! Additionally to the INCAR file, KPOINTS and POTCAR need to be supplied in the parent folder. Finally, run a VASP calculation in every subfolder within the "displacements" folder.

- Collect the results and calculate the Raman tensors via
```bash
python RamanPy -t
```
- calculate the Raman intensity and apply the smearing
```bash
python RamanPy -s
```
- plot the spectra
```bash
python RamanPy --plotRaman
```
<br><br>

## IR spectroscopy
- If the effective ionic charges are present in phonopy.yaml or OUTCAR, simply run
```bash
python RamanPy -IR --plotIR
```
- For a calculation of the reflectance, run
```bash
python RamanPy -R --plotReflectance
```
<br><br>

## Limitations
More options can be enabled by applying additional flags when executing RamanPy. Check
```bash
python RamanPy -h
```
for a complete set of flags. Note, that some options are only available from the python API. Two examplary calculations can be found in the examples subfolder.
<br>

- The phonon modes are ordered by their frequencies in ascending/descending order, depending on the file used to read the phonons!
- Symmetry analysis is only possible if FORCE_CONSTANTS are present. This allows to exclude silent modes and minimizes the numerical costs.
- Molecular point groups are not supported by the symmetry analysis.
- Only the intensity of TO modes can be calculated! Check the selection rules on which photon propagation directions are affected.
- The unit of the Raman tensors is not checked. Use arbitrary units for showcasing results (as is standard in literature).

<br>

## Cite
If you use this software for any of your projects, please consider citing one of the following:
- github
- https://doi.org/10.1002/pssa.202300968
- https://doi.org/10.1021/acs.jpcc.4c05225 
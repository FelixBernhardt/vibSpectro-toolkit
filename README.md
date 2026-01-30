
# Raman-spectroscopy

## Theory
The Raman tensor can be approximated as:<br><br>
$\alpha(\omega_0)\approx\dfrac{\partial \epsilon(\omega_0)}{\partial U}=\dfrac{\epsilon_{u^+}(\omega_0)-\epsilon_{u^-}(\omega_0)}{2\Delta u}$<br><br>
With $\omega_0$ being the laser wavelength, $\epsilon$ the dielectric function and $U$ a phonon eigenmode. The right hand side is a numerical implementation of the differential for ionic displacements in + and - direction of the phonon mode $U$ by a distance of $\Delta u$. As momentum has to be conserved, only phonons near Γ can contribute to Raman scattering (in a first approximation at least).<br><br>
The intensity can then be calculated as:<br><br>
$I\sim|\hat{e}_s\alpha\hat{e}_i|^2\dfrac{(\omega_0-\omega_p)^4}{\omega_p}(n+1)$<br><br>
With $n$ being the Bose-Einstein occupation number, $\omega_p$ the phonon frequency and $\hat{e}$ the polarization direction of the incident and scattered light respectively.<br><br><br>


## Phonon modes and frequencies at Γ-point
In order to start the script you need the phononic eigenmodes at Γ. Usage of the $phonopy$ format is recommended. The following files are needed:
- FORCE_CONSTANTS ?
- phonopy.yaml
- qpoints.yaml

Phonopy writes the $phonopy.yaml$ file per default. FORCE_CONSTANTS can be calculated from FORCE_SETS or different DFT calculators by
```bash
phonopy --writefc --dim="x y z"
```
where $x,y,z$ are the dimensions of the supercell used, if finite-displacement method is chosen. The dynamical matrix at Γ can be obtained by
```bash
phonopy --readfc --writedm --qpoints=\"0 0 0\"
```

Further information is written in the documentation at https://phonopy.github.io/phonopy/.
<br><br>

## Resonant Raman spectroscopy
- Prepare your structures by displacing the ions along the phonon eigenvector in plus and minus direction
```bash
python Ramanpy -d -m=<modelist>
```
This creates folders for all considered modes and displacements (only plus and minus direction to save numerical cost). Make sure to include the necessary files specified in the following in the parent folder where you run RamanPy.

- The electronic contribution to the dielectric function needs to be calculated for all created structurs. For VASP, a possible INCAR looks like this:
```bash
LOPTICS = .TRUE. # calculate the dielectric function as a sum over bands
NBANDS  = ...    # number of bands, check for convergence
EDIFF   = 1.e-8  # low value for accurate eigenvalues
ISMEAR  = 0      # Do not use -5 !
SIGMA   = 0.02   # depends on system, 0.02 should be fine
```
As for all optical calculations, check for k-point convergence! Additionally to the INCAR file, KPOINTS and POTCAR need to be supplied in the parent folder. Finally, run a VASP calculation in every subfolder within the "displacements" folder.

- Collect the results and calculate the Raman tensors via
```bash
python Ramanpy -t -m=<modelist>
```
- calculate the Raman intensity and apply the smearing
```bash
python Ramanpy -s -m=<modelist>
```
- plot the spectra
```bash
python Ramanpy -p
```
<br><br><br>

More option can be enabled by applying additional flags when executing RamanPy. Check
```bash
python Ramanpy -h
```
for a complete set of flags.

## Limitations
- Ensure to not include acoustic modes when applying the -nosym flag. For numerical reasons, they would dominate the resulting spectrum.
- Degeneracy is not completely integrated, the user has to set up the Raman tensor elements manually for the degenerate cases (but additional calculations are not necessary)
- Not all point groups are supported by the symmetry flags (I am too lazy...)
- Only the intensity of TO modes can be calculated! Check the selection rules on which photon propagation directions are affected (e.g. with the Bilbao Crystallographic Server).
- All cell and phonon information can also be read in from POSCAR and OUTCAR files by using the "-nosym -allVASP" flags when running RamanPy. This allows to bypass the phonopy interface and enables to calculate the Raman spectrum with some fixed atoms.
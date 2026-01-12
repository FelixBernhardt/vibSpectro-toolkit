
# Raman-spectroscopy

## Theory
The Raman tensor of a phonon mode $m$ can be approximated as:
<br><br>
$\alpha_m(\omega_0)\approx\dfrac{\partial \epsilon(\omega_0)}{\partial U}=\dfrac{\epsilon_{u^+}(\omega_0)-\epsilon_{u^-}(\omega_0)}{2\Delta u}$,
<br><br>
with $\omega_0$ being the laser wavelength, $\epsilon$ the dielectric function and $U$ the atomic displacements according to the phonon eigenmode $m$. The right hand side is a numerical implementation of the differential for ionic displacements in + and - direction of the phonon by a distance of $\Delta u$. As momentum has to be conserved, only phonons near Γ can contribute to Raman scattering.
<br><br>
The intensity can then be calculated as:
<br><br>
$I_m(\omega_0)\sim|\hat{e}_s\alpha\hat{e}_i|^2\dfrac{(\omega_0-\omega_m)^4}{\omega_m}(n+1)$,
<br><br>
with $n$ being the Bose-Einstein occupation number, $\omega_m$ the phonon frequency, and $\hat{e}$ the polarization direction of the incident and scattered light respectively.
<br><br><br>


## Phonon modes and frequencies at Γ-point
In order to start the script you need the phononic eigenmodes at Γ. Only $phonopy$'s format is supported! The following files are needed:
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

Further information on how to obtain phonon frequencies and eigenmodes at Γ is written in the documentation at https://phonopy.github.io/phonopy/.
<br><br>

## Resonant Raman spectroscopy
- Prepare your structures by displacing the ions along the phonon eigenvector in plus and minus direction
```bash
Ramanpy -d -m=<modelist>
```
This creates folders for all considered modes and displacements (only plus and minus direction, acoustic modes are ignored to save numerical cost).

- Afterwards, the electronic contribution to the dielectric function needs to be calculated for all created structurs. For VASP, a possible INCAR looks like this:
```bash
LOPTICS = .TRUE. # calculate the dielectric function as a sum over bands
NBANDS  = ...    # number of bands, check for convergence
EDIFF   = 1.e-8  # low value for accurate eigenvalues
ISMEAR  = 0      # should work universally
SIGMA   = ...    # depends on system
```
As for all optical calculations, check for k-point convergence!
- Collect the results and calculate the Raman tensors via
```bash
Ramanpy -t
```
- calculate the Raman intensity and apply the smearing
```bash
Ramanpy -s
```
- plot the spectra
```bash
Ramanpy -p
```
<br><br><br>
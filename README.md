
# VASP Raman-spectroscopy

## Theory
The Raman tensor can be approximated as:<br><br>
$\alpha(\omega_0)\approx\dfrac{\partial \epsilon(\omega_0)}{\partial U}=\dfrac{\epsilon_{u^+}(\omega_0)-\epsilon_{u^-}(\omega_0)}{2\Delta u}$<br><br>
With $\omega_0$ being the laser wavelength, $\epsilon$ the dielectric function and $U$ a phonon eigenmode. The right hand side is a numerical implementation of the differential for ionic displacements in + and - direction of the phonon mode $U$ by a distance of $\Delta u$. As momentum has to be conserved, only phonons near Γ can contribute to Raman scattering (in a first approximation at least).<br><br>
The intensity can then be calculated as:<br><br>
$I\sim|\hat{e}_s\alpha\hat{e}_i|^2\dfrac{(\omega_0-\omega_p)^4}{\omega_p}(n+1)$<br><br>
With $n$ being the Bose-Einstein occupation number, $\omega_p$ the phonon frequency and $\hat{e}$ the polarization direction of the incident and scattered light respectively.<br><br><br>


## Phonon modes and frequencies at Γ-point
- start with optimized structure (EDIFFG = -0.005 or lower)

- set the following tags in INCAR:
```bash
LREAL  = .FALSE. # always needed for accurate forces
IBRION = 5 or    # phonons at Γ with finite differences
IBRION = 7       # or DFPT (needs also NSW=1)
ISYM   = 0       # always symmetry off for phonons!
NWRITE = 3       # to print the "Eigenvectors after division by SQRT(mass)"
EDIFF  = 1.e-8   # low value for accurate forces
```
After the calculation you can visualize the modes with `Avogadro` by running `VASP2g98.py` (needs POSCAR and OUTCAR):
```bash
python VASP2g98.py
```
<br><br>

## Resonant Raman spectroscopy
prepare your structures by displacing the ions along the phonon eigenvector in plus and minus direction
- rename the previous OUTCAR to OUTCAR.phon, and the POSCAR to POSCAR.phon
- run
```bash
python VASP_Raman.py -g <modelist>
```
- run VASP for all the created POSCARs using the following tags in INCAR:
```bash
LOPTICS = .TRUE. # calculate the dielectric function as a sum over bands
NBANDS  = ...    # number of bands, check for convergence
EDIFF   = 1.e-8  # low value for accurate eigenvalues
ISMEAR  = 0      # Do not use -5 !
SIGMA   = 0.02   # depends on system, 0.02 should be fine
```
As for all optical calculations, check for k-point convergence!
- collect the results and calculate the Raman tensors
```bash
python VASP_Raman.py -c <modelist>
```
- calculate the Raman intensity
```bash
python VASP_Raman.py -s <modelist>
```
check the information provided by the script
```bash
python VASP_Raman.py -h
```
<br><br><br>

## Contributors
Felix Bernhardt (JLU Gießen)

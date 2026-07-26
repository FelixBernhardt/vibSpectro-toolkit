# Vibrational Spectroscopy: Theory

## Infrared spectroscopy
Infrared spectra are calculated via the effective charges $Z_a$ of ion $a$ and the phononic eigenvectors $\hat{Q}$. The imaginary part of the ionic contribution to the dielectric function (i.e. phononic absorption) is calculated by:<br><br>
$\Im(\epsilon_{ij}(\omega_p))=\frac{1}{4\pi^2\epsilon_0V}\left(\sum\limits_{a,k}Z_{a,ik}\hat{Q}_{p,a}\right)\left(\sum\limits_{a,k}Z_{a,jk}\hat{Q}_{p,a}\right).$<br><br>
The ionic contribution of the imaginary part of the dielectric function thus yields delta-peaks at the phonon frequencies $\omega_p$. A Lorentzian smearing is again applied to obtain a more realistic spectrum.
The corresponding real part is calculated using Kramers-Kronig relations:<br><br>
$\Re(\epsilon_{ij}(\omega))=\sum\limits_p\frac{\Im(\epsilon_{ij}(\omega_p))(\omega_p^2-\omega^2)}{(\omega_p^2-\omega^2)^2+\gamma^2\omega^2}$.<br><br>
The reflectance $R$ in the infrared regime can then be calculated by:<br><br>
$R_{ij}(\omega)=\left(\frac{n-1}{n+1}\right)^2, \quad\quad n=\sqrt{\Re(\epsilon_{ij}(\omega))+i\Im(\epsilon_{ij}(\omega))}$

## Raman spectroscopy
The first order Raman tensor $\alpha_p$ of phonon mode $p$ is approximated as:<br><br>
$\alpha_p(\omega_0)\approx\dfrac{\partial \epsilon(\omega_0)}{\partial Q}=\dfrac{\epsilon_{q^+}(\omega_0)-\epsilon_{q^-}(\omega_0)}{2\Delta q},$<br><br>
with $\omega_0$ being the laser wavelength, $\epsilon$ the dielectric function and $Q$ a phonon eigenmode. The right hand side is a numerical implementation of the differential for ionic displacements in + and - direction of the phonon mode $Q$ by a distance of $\Delta q$. As momentum has to be conserved, only phonons near $\Gamma$ can contribute to Raman scattering (in a first approximation at least).<br><br>
The Stokes intensity $I_p$ of phonon mode $p$ can then be calculated as:<br><br>
$I_p\sim|\hat{e}_s\alpha\hat{e}_i|^2\dfrac{(\omega_0-\omega_p)^4}{\omega_p}(n+1),$<br><br>
with $n$ being the Bose-Einstein occupation number, $\omega_p$ the phonon frequency and $\hat{e}$ the polarization direction of the incident and scattered light respectively. Both resonant and non-resonant Raman spectra can be obtained in this formulation (again, as a first approximation).
The anti-Stokes intensity is instead calculated as:<br><br>
$I_p\sim|\hat{e}_s\alpha\hat{e}_i|^2\dfrac{(\omega_0+\omega_p)^4}{\omega_p}n$.
<br><br>
All spectra are smeared with a Lorentzian smearing of the form:<br><br>
$I(\omega)=\sum\limits_p I_p \gamma / ([\omega_p-\omega_0]^2+\gamma^2 ),$<br><br>
with an arbitrary smearing width $\gamma$.

## Handling of LO phonon modes
LO frequencies and eigenvectors can be computed within phonopy using the non-analytical term correction tag for different directions. Here, the LO eigenvectors are, in a first approximation, considered to be identical to their TO counterparts. This counterpart is determined by the scalar product of the eigenvectors $Q$:<br><br>
$\braket{Q_{LO}|Q_{TO}}.$<br><br>
Then, the TO phonon frequencies are modified to their corresponding LO phonon frequencies, while the Raman tensor is not modified.


## Symmetry considerations
Not all Raman tensors of all phonon modes need to be calculated: It is sufficient to only include phonon modes that are Raman active according to group symmetry. The phonon mode symmetry analysis is only available when using $phonopy$ (more specifically, the FORCE_CONSTANTS file has to be present), and relies on the formulations used therein. Further, only one mode per pair/triplett of degenerate modes needs to be explicitly calculated, since the Raman tensor of a degenerate mode can be constructed from one of its degenerate partners. Finally, purely rotational modes (only for molecules), as well as acoustic phonon modes, can be excluded from the calculations. The latter consideration is always applied for all calculations.

## Phonon modes and frequencies at Γ-point
In order to start the script you need the phononic eigenmodes at Γ. Usage of the $phonopy$ format is recommended, but VASP and QuantumEspresso are supported as well. The following files are needed:
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

For QuantumEspresso, check the example python script for a full workflow.

<br><br>

# Running calculations
Check the examples folder, or the helper function of the command-line interface script. Note, that some options are only available from the python API.

<!--
## Raman spectroscopy
- Prepare your structures by displacing the ions along the phonon eigenvector in plus and minus direction, for each phonon mode
```bash
python vibSpectro-toolkit -d
```
This creates folders for all considered modes and displacements (only one folder for each direction, i.e. two folder per mode). As a default, silent modes are ignored. Make sure to include the necessary files specified in the following in the parent folder where you run vibSpectro-toolkit.

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
python vibSpectro-toolkit -t
```
- calculate the Raman intensity and apply the smearing
```bash
python vibSpectro-toolkit -s
```
- plot the spectra
```bash
python vibSpectro-toolkit --plotRaman
```
<br><br>

## IR spectroscopy
- If the effective ionic charges are present in phonopy.yaml or OUTCAR, simply run
```bash
python vibSpectro-toolkit -IR --plotIR
```
- For a calculation of the reflectance, run
```bash
python vibSpectro-toolkit -R --plotReflectance
```
<br><br>

More options can be enabled by applying additional flags when executing vibSpectro-toolkit. Check
```bash
python vibSpectro-toolkit -h
```
for a complete set of flags. Note, that some options are only available from the python API. Two examplary calculations can be found in the examples subfolder.
<br>
-->

## Limitations

- The phonon modes are ordered by their frequencies in ascending/descending order, depending on the file used to read the phonons!
- Symmetry analysis is only possible if FORCE_CONSTANTS are present. This allows to exclude silent modes and minimizes the numerical costs.
- Molecular point groups are not supported by the symmetry analysis.
- The unit of the Raman tensors is not checked. Use arbitrary units for showcasing results (as is standard in literature).
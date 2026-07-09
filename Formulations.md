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
The Raman tensor $\alpha_p$ of phonon mode $p$ is approximated as:<br><br>
$\alpha_p(\omega_0)\approx\dfrac{\partial \epsilon(\omega_0)}{\partial Q}=\dfrac{\epsilon_{q^+}(\omega_0)-\epsilon_{q^-}(\omega_0)}{2\Delta q},$<br><br>
with $\omega_0$ being the laser wavelength, $\epsilon$ the dielectric function and $Q$ a phonon eigenmode. The right hand side is a numerical implementation of the differential for ionic displacements in + and - direction of the phonon mode $Q$ by a distance of $\Delta q$. As momentum has to be conserved, only phonons near $\Gamma$ can contribute to Raman scattering (in a first approximation at least).<br><br>
The Stokes intensity $I_p$ of phonon mode $p$ can then be calculated as:<br><br>
$I_p\sim|\hat{e}_s\alpha\hat{e}_i|^2\dfrac{(\omega_0-\omega_p)^4}{\omega_p}(n+1),$<br><br>
with $n$ being the Bose-Einstein occupation number, $\omega_p$ the phonon frequency and $\hat{e}$ the polarization direction of the incident and scattered light respectively. Both resonant and non-resonant Raman spectra can be obtained in this formulation.
The anti-Stokes intensity is instead calculated as:<br><br>
$I_p\sim|\hat{e}_s\alpha\hat{e}_i|^2\dfrac{(\omega_0+\omega_p)^4}{\omega_p}n$.
<br><br>
All spectra are smeared with a Lorentzian smearing of the form:<br><br>
$I(\omega)=\sum\limits_p I_p \gamma / ([\omega_p-\omega_0]^2+\gamma^2 ),$<br><br>
with an arbitrary smearing width $\gamma$.

## Handling of LO phonon modes
LO frequencies and eigenvectors can be computed within phonopy using the non-analytical term correction tag for different directions. Here, the LO eigenvectors are, in a first approximation, considered to be identical to theit TO counterparts. This counterpart is determined by the scalar product of the eigenvectors $Q$:<br><br>
$\braket{Q_{LO}|Q_{TO}}.$<br><br>
Then, the TO phonon frequencies are modified to their corresponding LO phonon frequencies, while the Raman tensor is not modified.


## Symmetry considerations
Not all Raman tensors of all phonon modes need to be calculated: It is sufficient to only include phonon modes that are Raman active according to group symmetry. The phonon mode symmetry analysis is only available when using $phonopy$ (more specifically, the FORCE_CONSTANTS file has to be present), and relies on the formulations used therein. Further, only one mode per pair/triplett of degenerate modes needs to be explicitly calculated, since the Raman tensor of a degenerate mode can be constructed from one of its degenerate partners. Finally, acoustic phonon modes, as well as purely rotational modes (only for molecules), can be excluded from the calculations. The latter consideration is always applied for all calculations.
# Theory used by this implementation

The implementation follows the paper notation throughout.

## System model

\[
\mathbf Y=\mathbf\Theta\mathbf Z+\mathbf W,
\qquad
\mathbf Z=\operatorname{diag}(\mathbf a)\mathbf H^T,
\]

where \(\mathbf\Theta\in\mathbb C^{L\times N}\), \(\mathbf Z\in\mathbb C^{N\times M}\), and \(\mathbf Y\in\mathbb C^{L\times M}\).

For MTD \(i\),

\[
\mathbf z_i\mid\gamma_i\sim\mathcal{CN}(\mathbf0,\gamma_i\mathbf I_M).
\]

For a proper complex Gaussian, the expected SBL term is

\[
\mathbb E[\log p(\mathbf Z\mid\boldsymbol\gamma)]
\propto
-M\sum_{i=1}^{N}
\left[
\log\gamma_i+\frac{\eta_i}{\gamma_i}
\right],
\]

with

\[
\eta_i=\Sigma_{ii}+\frac{\|\boldsymbol\mu_i\|_2^2}{M}.
\]

The old real-Gaussian \(1/2\) factor is therefore not used.

## Correlation-aware term

The ANC construction is unchanged:

\[
\mathbf\Omega=\alpha\left(\beta\mathbf1-\mathbf C\right).
\]

The correlation term is

\[
-\frac12\boldsymbol\gamma^T\mathbf\Omega\boldsymbol\gamma.
\]

Combining it with the correct complex-MMV SBL term gives

\[
\frac1{\gamma_i}
-\frac{\eta_i}{\gamma_i^2}
+\frac1M(\mathbf\Omega\boldsymbol\gamma)_i=0.
\]

The lagged interaction used by CA-SBL is therefore

\[
\phi_i^{[k]}
=\frac1M
(\mathbf\Omega\boldsymbol\gamma^{[k-1]})_i.
\]

For \(\phi_i>0\), the numerically stable equivalent of the positive quadratic root is used:

\[
\boxed{
\gamma_i^{[k]}
=
\frac{2\eta_i^{[k]}}
{\sqrt{1+4\phi_i^{[k]}\eta_i^{[k]}}+1}
}.
\]

For \(\phi_i\le0\), the implementation uses the conventional SBL fallback
\(\gamma_i^{[k]}=\eta_i^{[k]}\). No upper clipping at 1 is applied.

## Optimized E-step

\[
\mathbf\Pi
=\mathbf\Theta\operatorname{diag}(\boldsymbol\gamma)\mathbf\Theta^H
+\sigma^2\mathbf I_L.
\]

The code factorizes the \(L\times L\) matrix \(\mathbf\Pi\) with Cholesky instead of calling an explicit inverse. It computes

\[
\boldsymbol\mu
=\operatorname{diag}(\boldsymbol\gamma)\mathbf\Theta^H\mathbf\Pi^{-1}\mathbf Y,
\]

and only the required diagonal

\[
\Sigma_{ii}
=\gamma_i-
\gamma_i^2
[\mathbf\Theta^H\mathbf\Pi^{-1}\mathbf\Theta]_{ii}.
\]

This avoids constructing the full \(N\times N\) posterior covariance in normal runs.

## Important interpretation note

Because the implementation does not constrain \(\boldsymbol\gamma\) to \([0,1]^N\), and the ANC construction does not guarantee that \(\mathbf\Omega\) is positive semidefinite, the quadratic expression should not automatically be described as a normalizable Gaussian prior over the whole nonnegative orthant. The code therefore treats it as the correlation-aware quadratic term in the MAP/EM surrogate. If the manuscript later claims a proper probability density \(p(\boldsymbol\gamma\mid\mathbf\Omega)\), its support/normalizability conditions should be stated separately.

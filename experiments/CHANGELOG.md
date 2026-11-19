# Empirical Study Protocol & Experiment Changelog

This changelog records the design protocol, frozen configurations, and rationale for all experiments (E1–E6) in `hesse`. In accordance with rigorous scientific practice, hyperparameter choices and evaluation protocol are frozen a priori. Any adjustments made during execution and their motivations are documented here.

---

## 1. Experimental Protocol (Frozen Prior to Execution)

### 1.1 Multi-Seed Replication
- Every configuration is executed across **3 independent random seeds**: $S \in \{42, 43, 44\}$.
- All reported metrics specify sample mean and sample standard deviation ($\mu \pm \sigma$).

### 1.2 Fixed Curvature Evaluation Sets
- In all experiments, curvature metrics ($\lambda_{\max}$, top-$k$ eigenvalues, trace, preconditioned sharpness, and sign-directional curvature) are evaluated on a **strictly fixed evaluation batch** (or full-batch tensor) so that curvature measurements are strictly comparable across training iterations.

### 1.3 Standardized Metric Definitions
1. **Time to Threshold ($t_{\text{threshold}}$):**
   - The first discrete training step $t$ at which sharpness exceeds the theoretical linear stability boundary ($\lambda_{\max} \ge 2/\eta$ for GD/SGD, or preconditioned sharpness $\ge \frac{2+2\beta_1}{(1-\beta_1)\eta}$ for Adam).
   - If the threshold is never crossed during training, this value is recorded as `null` / `N/A`.
2. **Plateau Window ($W_{\text{plateau}}$):**
   - If $t_{\text{threshold}}$ is reached: defined as all logged evaluation steps $t \ge t_{\text{threshold}}$ through the end of training.
   - If the threshold is never reached (or is mathematically undefined, e.g., for Lion): defined as the final $40\%$ of training steps ($t \ge 0.6 \times t_{\text{final}}$).
3. **Plateau Sharpness Ratio ($R_{\text{plateau}}$):**
   - The arithmetic mean of the normalized curvature ratio $\lambda_{\max}(H_t) / \text{Threshold}$ evaluated over all points in the plateau window.
   - Values $R_{\text{plateau}} \approx 1.0$ indicate exact edge-of-stability hovering.
4. **Non-Monotone Loss Fraction ($f_{\text{non-mono}}$):**
   - The fraction of consecutive logged training steps where the empirical loss increased:
     $$f_{\text{non-mono}} = \frac{1}{K - 1} \sum_{i=2}^K \mathbf{1}\left[\mathcal{L}(\theta_{t_i}) > \mathcal{L}(\theta_{t_{i-1}})\right]$$
   - A non-zero fraction during stable convergence directly diagnoses non-monotonic descent characteristic of the edge of stability.
5. **Measured Runtime:**
   - Wall-clock execution time (in seconds) measured for each experiment.

---

## 2. Experiment Matrix & Hypotheses

| ID | Title | Optimizer & Config | Primary Question / Expected Result |
| :--- | :--- | :--- | :--- |
| **E1** | GD Edge of Stability | Full-batch GD, tanh MLP, MSE loss, 4 learning rates $\eta \in \{0.01, 0.02, 0.05, 0.10\}$ | Sharpness $\lambda_{\max}$ undergoes progressive sharpening and hovers near $2/\eta$, accompanied by non-monotonic loss descent. |
| **E2** | SGD Batch Size Scaling | Minibatch SGD, batch sizes $B \in \{32, 128, 512, \text{full}\}$ at $\eta = 0.05$ | Full and large batch sizes hover at $2/\eta = 40.0$; small batch sizes dissipate curvature via stochastic gradient noise, suppressing $\lambda_{\max} < 2/\eta$. |
| **E3** | Adam Preconditioned EoS | Adam ($\beta_1 \in \{0.9, 0.0\}$), 3 learning rates $\eta \in \{10^{-3}, 3 \times 10^{-3}, 10^{-2}\}$ | Raw $\lambda_{\max}$ does not hover at $2/\eta$; preconditioned sharpness $\lambda_{\max}(P_t^{-1} H)$ tracks the theoretical $(2+2\beta_1)/((1-\beta_1)\eta)$ bound. |
| **E4** | Lion Sign-Update Geometry | Lion optimizer, 3 learning rates $\eta \in \{10^{-4}, 3 \times 10^{-4}, 10^{-3}\}$ | Open question: Does Lion exhibit an eigenvalue plateau? Directional curvature along $\operatorname{sign}(m_t)$ is evaluated to test coordinate-wise saturation vs $L_2$ bounds. |
| **E5** | Architectural Curvature | MLP vs Small CNN vs Tiny Transformer under GD and Adam | Evaluates whether EoS hovering and spectral gaps ($\lambda_1 - \lambda_2$) differ across standard architectures at matched loss. |
| **E6** | Loss Function Geometry | Full-batch GD ($\eta = 0.04$), MSE vs Categorical Cross-Entropy | Tests whether MSE maintains persistent EoS hovering while Cross-Entropy exhibits late-stage flattening caused by logit divergence. |

---

## 3. Log of Protocol Modifications & System Adaptations

1. **Transformer Autograd Backend Adaptation (PyTorch CPU Second-Derivative Support):**
   - *Observation:* PyTorch CPU's native scaled dot-product attention (`aten::_scaled_dot_product_flash_attention_for_cpu`) does not support double-backward (`grad` of `grad`), causing runtime errors during Hessian-vector product calculation.
   - *Resolution:* Wrapped `TinyTransformer.forward()` within `torch.nn.attention.sdpa_kernel(SDPBackend.MATH)` to force pure-math attention whose computation graph is infinitely differentiable by autograd.

2. **CPU Runtime Optimization for Architectural Sweep (E5):**
   - *Observation:* Evaluating 40 Hutchinson trace samples and 5 Lanczos eigenvalues every 2 steps over 500 images on a 3-layer CNN required excessive compute on laptop CPU.
   - *Resolution:* Made `trace_samples` configurable in `TrackerConfig` (set to 15 in E5), reduced dataset subset size to 200 samples, standardized CNN channels to `[16, 32]`, ViT to `patch_size: 8, embed_dim: 32`, and set tracking interval to 4 steps over 35 steps. This maintained full spectral comparability across all 3 architectures while completing the 18-run sweep in 396s.

3. **Lion Curvature Instrumentation:**
   - Exposed `Curvature.hvp(v)` to compute sign-aware directional curvature $\kappa_{\text{sign}} = \frac{u_t^T H_t u_t}{\|u_t\|_2^2}$ along the update vector $u_t = \operatorname{sign}(\beta_1 m_t + (1-\beta_1) g_t)$.

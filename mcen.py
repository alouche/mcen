import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import norm
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
import warnings
import pickle
from itertools import product


@dataclass
class IdentifiabilityCondition:
    """Represents formal identifiability conditions for MCHL"""

    condition_name: str
    is_satisfied: bool
    test_statistic: float
    p_value: float
    critical_threshold: float
    violation_reason: Optional[str] = None


@dataclass
class ConvolutionalKernel:
    """Represents convolutional propagation kernel"""

    kernel_type: str  # 'gaussian', 'exponential', 'power_law'
    support_width: int
    parameters: Dict[str, float]
    normalization_constant: float
    _discrete_vals: Optional[np.ndarray] = None
    _discrete_lags: Optional[np.ndarray] = None

    def evaluate(self, lags: np.ndarray) -> np.ndarray:
        if self.kernel_type == "gaussian":
            sigma = self.parameters["sigma"]
            return (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(
                -0.5 * (lags / sigma) ** 2
            )
        elif self.kernel_type == "exponential":
            lambda_param = self.parameters["lambda"]
            return np.exp(-lambda_param * lags) * lambda_param
        elif self.kernel_type == "power_law":
            alpha = self.parameters["alpha"]
            kappa = self.parameters["kappa"]
            return (alpha / kappa) * (1 + lags / kappa) ** (-alpha - 1)
        else:
            raise ValueError(f"Unknown kernel type: {self.kernel_type}")

    def build_discrete(self, max_memory: int, retention: float = 0.99) -> None:
        Hmax = min(self.support_width, max_memory)
        lags = np.arange(1, Hmax + 1)
        vals = self.evaluate(lags)
        s = vals.sum()
        vals = vals / s if s > 0 else np.zeros_like(lags, dtype=float)
        self._discrete_vals = vals
        self._discrete_lags = lags


@dataclass
class MCHLResult:
    """MCHL results with theoretical guarantees"""

    intervention: str
    state_cluster: Optional[int]
    decay_type: str
    parameters: Dict[str, Any]
    half_life: float
    r2_in_sample: float
    effect_curve: np.ndarray
    time_lags: np.ndarray
    compression_threshold: float = 0.01
    identifiability_conditions: List[IdentifiabilityCondition] = field(
        default_factory=list
    )
    fisher_information_matrix: Optional[np.ndarray] = None
    asymptotic_variance: Optional[np.ndarray] = None
    tail_bound: Optional[float] = None
    convergence_rate: Optional[float] = None
    effect_mass: Optional[float] = None
    snr: Optional[float] = None
    mass_half_life: Optional[float] = None
    posterior_prob: Optional[float] = None


class TheoreticalDecayModel:
    """Base class for decay models with theoretical properties."""

    def __init__(self, name: str):
        self.name = name
        self.parameter_bounds: Dict[str, Tuple[float, float]] = {}

    def function(self, k: np.ndarray, params: Dict[str, float]) -> np.ndarray:
        raise NotImplementedError

    def gradient(
        self, k: np.ndarray, params: Dict[str, float]
    ) -> Dict[str, np.ndarray]:
        raise NotImplementedError

    def fisher_information(
        self, k: np.ndarray, params: Dict[str, float], noise_var: float = 1.0
    ) -> np.ndarray:
        grad = self.gradient(k, params)
        param_names = list(grad.keys())
        n_params = len(param_names)
        fim = np.zeros((n_params, n_params))
        for i, pi in enumerate(param_names):
            for j, pj in enumerate(param_names):
                fim[i, j] = np.sum(grad[pi] * grad[pj]) / max(noise_var, 1e-8)
        return fim

    def check_identifiability(
        self, k: np.ndarray, params: Dict[str, float]
    ) -> List[IdentifiabilityCondition]:
        conditions = []
        try:
            fim = self.fisher_information(k, params)
            rank = int(np.linalg.matrix_rank(fim))
            full_rank = rank == fim.shape[0]
            conditions.append(
                IdentifiabilityCondition(
                    condition_name="Fisher_Information_Full_Rank",
                    is_satisfied=full_rank,
                    test_statistic=float(rank),
                    p_value=0.0,
                    critical_threshold=float(fim.shape[0]),
                    violation_reason=None if full_rank else "FIM rank deficient",
                )
            )
            cond = float(np.linalg.cond(fim)) if fim.size > 0 else np.inf
            well_conditioned = cond < 1e3
            conditions.append(
                IdentifiabilityCondition(
                    condition_name="Fisher_Information_Condition_Number",
                    is_satisfied=well_conditioned,
                    test_statistic=cond,
                    p_value=0.0,
                    critical_threshold=1e3,
                    violation_reason=None
                    if well_conditioned
                    else "FIM ill-conditioned",
                )
            )
        except Exception as e:
            for name in [
                "Fisher_Information_Full_Rank",
                "Fisher_Information_Condition_Number",
            ]:
                conditions.append(
                    IdentifiabilityCondition(
                        condition_name=name,
                        is_satisfied=False,
                        test_statistic=0.0 if "Rank" in name else float("inf"),
                        p_value=1.0,
                        critical_threshold=1.0 if "Rank" in name else 1e3,
                        violation_reason=f"Error computing FIM: {e}",
                    )
                )

        try:
            grad = self.gradient(k, params)
            G = np.array([grad[p] for p in grad.keys()])
            if G.ndim == 2:
                gram = G @ G.T
                gram_rank = int(np.linalg.matrix_rank(gram))
                separable = gram_rank == G.shape[0]
                crit = float(G.shape[0])
            else:
                gram_rank = 1
                separable = True
                crit = 1.0
            conditions.append(
                IdentifiabilityCondition(
                    condition_name="Parameter_Separability",
                    is_satisfied=separable,
                    test_statistic=float(gram_rank),
                    p_value=0.0,
                    critical_threshold=crit,
                    violation_reason=None
                    if separable
                    else "Gradients not linearly independent",
                )
            )
        except Exception as e:
            conditions.append(
                IdentifiabilityCondition(
                    condition_name="Parameter_Separability",
                    is_satisfied=False,
                    test_statistic=0.0,
                    p_value=1.0,
                    critical_threshold=1.0,
                    violation_reason=f"Error checking separability: {e}",
                )
            )
        return conditions


class ExponentialDecay(TheoreticalDecayModel):
    def __init__(self):
        super().__init__("exponential")
        self.parameter_bounds = {"lambda": (0.001, 10.0)}

    def function(self, k: np.ndarray, params: Dict[str, float]) -> np.ndarray:
        return np.exp(-params["lambda"] * k)

    def gradient(
        self, k: np.ndarray, params: Dict[str, float]
    ) -> Dict[str, np.ndarray]:
        exp_term = np.exp(-params["lambda"] * k)
        return {"lambda": -k * exp_term}

    def half_life(self, params: Dict[str, float]) -> float:
        return float(np.log(2) / params["lambda"])


class PowerLawDecay(TheoreticalDecayModel):
    def __init__(self):
        super().__init__("power_law")
        self.parameter_bounds = {"lambda": (0.1, 5.0), "kappa": (0.1, 100.0)}

    def function(self, k: np.ndarray, params: Dict[str, float]) -> np.ndarray:
        return (1 + k / params["kappa"]) ** (-params["lambda"])

    def gradient(
        self, k: np.ndarray, params: Dict[str, float]
    ) -> Dict[str, np.ndarray]:
        base = 1 + k / params["kappa"]
        decay = base ** (-params["lambda"])
        grad_lambda = -np.log(base) * decay
        grad_kappa = (params["lambda"] * k / (params["kappa"] ** 2)) * decay / base
        return {"lambda": grad_lambda, "kappa": grad_kappa}

    def half_life(self, params: Dict[str, float]) -> float:
        return float(params["kappa"] * (2 ** (1 / params["lambda"]) - 1))


class BuildupPowerDecay(TheoreticalDecayModel):
    def __init__(self):
        super().__init__("buildup_power")
        self.parameter_bounds = {
            "lambda_rise": (0.01, 2.0),
            "lambda_decay": (0.1, 5.0),
            "kappa": (0.1, 100.0),
        }

    def function(self, k: np.ndarray, params: Dict[str, float]) -> np.ndarray:
        rise = 1.0 - np.exp(-params["lambda_rise"] * k)
        tail = (1.0 + k / params["kappa"]) ** (-params["lambda_decay"])
        return rise * tail

    def gradient(
        self, k: np.ndarray, params: Dict[str, float]
    ) -> Dict[str, np.ndarray]:
        exp_term = np.exp(-params["lambda_rise"] * k)
        rise = 1.0 - exp_term
        base = 1.0 + k / params["kappa"]
        tail = base ** (-params["lambda_decay"])
        d_lambda_rise = k * exp_term * tail
        d_lambda_decay = -np.log(base) * rise * tail
        d_kappa = (
            (params["lambda_decay"] * k / (params["kappa"] ** 2)) * (rise * tail) / base
        )
        return {
            "lambda_rise": d_lambda_rise,
            "lambda_decay": d_lambda_decay,
            "kappa": d_kappa,
        }

    def half_life(self, params: Dict[str, float]) -> float:
        def f(k_val):
            return (1.0 - np.exp(-params["lambda_rise"] * k_val)) * (
                1.0 + k_val / params["kappa"]
            ) ** (-params["lambda_decay"])

        K = np.linspace(0, 200, 2001)
        y = f(K)
        k_peak = float(K[np.argmax(y)])
        peak = float(np.max(y))
        if peak <= 0 or not np.isfinite(peak):
            return float("inf")
        target = 0.5 * peak
        tail_y = y[K >= k_peak]
        tail_k = K[K >= k_peak]
        idx = np.searchsorted(tail_y[::-1], target)
        if idx <= 0 or idx >= len(tail_y):
            return float("inf")
        return float(tail_k[len(tail_y) - idx - 1])


def _gaussian_kernel(u: np.ndarray) -> np.ndarray:
    return np.exp(-0.5 * u**2) / np.sqrt(2 * np.pi)


def _local_linear_smoother(
    x: np.ndarray,
    y: np.ndarray,
    x_eval: np.ndarray,
    h: float,
    w: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    x, y, x_eval = np.asarray(x, float), np.asarray(y, float), np.asarray(x_eval, float)
    n = len(x)
    w = np.ones(n, float) if w is None else np.asarray(w, float)
    h = max(float(h), 1e-6)
    yhat = np.empty_like(x_eval, dtype=float)
    sdiag = np.empty_like(x_eval, dtype=float)
    for i, x0 in enumerate(x_eval):
        u = (x - x0) / h
        K = _gaussian_kernel(u)
        Kw = K * w
        S0, S1, S2 = np.sum(Kw), np.sum(Kw * (x - x0)), np.sum(Kw * (x - x0) ** 2)
        denom = S0 * S2 - S1**2
        if denom <= 1e-12:
            wloc = Kw / max(S0, 1e-12)
            yhat[i] = np.sum(wloc * y)
            sdiag[i] = np.max(wloc)
            continue
        a = (S2 * np.sum(Kw * y) - S1 * np.sum(Kw * (x - x0) * y)) / denom
        yhat[i] = a
        wi = Kw * (S2 - (x - x0) * S1) / max(denom, 1e-12)
        sdiag[i] = np.sum(wi**2)
    return yhat, sdiag


def _kfold_cv_score(
    x: np.ndarray, y: np.ndarray, w: Optional[np.ndarray], h: float, k: int = 5
) -> float:
    n = len(x)
    idx = np.arange(n)
    folds = np.array_split(idx, k)
    errs = []
    for f in folds:
        tr = np.setdiff1d(idx, f, assume_unique=False)
        yhat, _ = _local_linear_smoother(
            x[tr], y[tr], x[f], h, None if w is None else w[tr]
        )
        errs.append(np.mean((y[f] - yhat) ** 2))
    return float(np.mean(errs))


def _moving_block_bootstrap(
    x: np.ndarray,
    y: np.ndarray,
    w: Optional[np.ndarray],
    x_eval: np.ndarray,
    h: float,
    B: int = 200,
    block_len: Optional[int] = None,
    alpha: float = 0.05,
) -> Tuple[np.ndarray, np.ndarray]:
    n = len(x)
    block_len = max(2, int(2 * n ** (1 / 3))) if block_len is None else block_len
    mu, _ = _local_linear_smoother(x, y, x, h, w)
    resid = y - mu
    preds = np.empty((B, len(x_eval)), dtype=float)
    for b in range(B):
        idx = []
        while len(idx) < n:
            start = np.random.randint(0, n - block_len + 1)
            idx.extend(range(start, start + block_len))
        idx = np.array(idx[:n])
        xb, yb = x[idx], mu[idx] + resid[idx]
        wb = None if w is None else w[idx]
        phat, _ = _local_linear_smoother(xb, yb, x_eval, h, wb)
        preds[b, :] = phat
    lo = np.quantile(preds, q=alpha / 2.0, axis=0)
    hi = np.quantile(preds, q=1 - alpha / 2.0, axis=0)
    return lo, hi


class NonparametricKernelDecay(TheoreticalDecayModel):
    def __init__(
        self, bandwidth_grid: Optional[List[float]] = None, allow_k0: bool = True
    ):
        super().__init__("nonparametric")
        self.allow_k0 = allow_k0
        self.bandwidth_grid = bandwidth_grid

    def function(self, k: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
        h = float(params["bandwidth"])
        x_train = np.asarray(params["x_train"], float)
        y_train = np.asarray(params["y_train"], float)
        w_train = params.get("w_train", None)
        yhat, _ = _local_linear_smoother(
            x_train, y_train, np.asarray(k, float), h, w_train
        )
        return yhat

    def gradient(self, k: np.ndarray, params: Dict[str, Any]) -> Dict[str, np.ndarray]:
        return {"bandwidth": np.zeros_like(np.asarray(k, float))}

    def fisher_information(self, *args, **kwargs) -> np.ndarray:
        return np.zeros((1, 1))

    def check_identifiability(
        self, k: np.ndarray, params: Dict[str, Any]
    ) -> List[IdentifiabilityCondition]:
        x_train = np.asarray(params["x_train"], float)
        y_train = np.asarray(params["y_train"], float)
        w_train = params.get("w_train", None)
        h = float(params["bandwidth"])
        _, sdiag = _local_linear_smoother(x_train, y_train, x_train, h, w_train)
        df = float(np.sum(sdiag))
        n = float(len(x_train))
        dof_ok = (df >= 2.0) and (df <= 0.7 * n)
        conds = [
            IdentifiabilityCondition(
                condition_name="Effective_DoF_Within_Range",
                is_satisfied=bool(dof_ok),
                test_statistic=df,
                p_value=0.0,
                critical_threshold=0.7 * n,
                violation_reason=None if dof_ok else "DoF too small/large",
            )
        ]
        hs = [max(1e-6, 0.8 * h), h, 1.2 * h]
        risks = [_kfold_cv_score(x_train, y_train, w_train, hh, k=5) for hh in hs]
        base = risks[1]
        stab_ok = (max(risks) / max(base, 1e-12)) <= 1.25
        conds.append(
            IdentifiabilityCondition(
                condition_name="CV_Risk_Stability",
                is_satisfied=bool(stab_ok),
                test_statistic=float(max(risks) / max(base, 1e-12)),
                p_value=0.0,
                critical_threshold=1.15,
                violation_reason=None if stab_ok else "Risk unstable vs bandwidth",
            )
        )
        return conds

    def half_life(self, params: Dict[str, Any]) -> float:
        h = float(params["bandwidth"])
        x_train = np.asarray(params["x_train"], float)
        y_train = np.asarray(params["y_train"], float)
        w_train = params.get("w_train", None)
        grid = np.linspace(0, max(1.0, float(x_train.max()) * 1.5), 500)
        yhat, _ = _local_linear_smoother(x_train, y_train, grid, h, w_train)
        peak_idx = int(np.argmax(yhat))
        peak = float(yhat[peak_idx])
        if not np.isfinite(peak) or peak <= 0:
            return float("inf")
        target = 0.5 * peak
        below = np.where(yhat[peak_idx:] <= target)[0]
        return float(grid[peak_idx + below[0]]) if len(below) > 0 else float("inf")


def mass_half_life(time_lags: np.ndarray, effect_curve: np.ndarray) -> float:
    order = np.argsort(time_lags)
    k = np.asarray(time_lags)[order].astype(int)
    tau = np.abs(np.asarray(effect_curve)[order])
    total = tau.sum()
    if total <= 0:
        return float("inf")
    cum = np.cumsum(tau)
    idx = int(np.searchsorted(cum, 0.5 * total))
    idx = min(idx, len(k) - 1)
    return float(k[idx])


class PrincipledInformationMetric:
    def __init__(self, metric_type: str = "jsd"):
        self.metric_type = metric_type

    @staticmethod
    def _safe_hist(x: np.ndarray, bins: int = 32) -> np.ndarray:
        x = np.asarray(x).ravel()
        if len(x) < 3 or np.allclose(np.std(x), 0):
            return np.ones(bins) / bins
        hist, _ = np.histogram(x, bins=bins, density=True)
        p = (hist + 1e-12) / (hist + 1e-12).sum()
        return p

    @staticmethod
    def js_divergence(x: np.ndarray, y: np.ndarray, bins: int = 32) -> float:
        p = PrincipledInformationMetric._safe_hist(x, bins)
        q = PrincipledInformationMetric._safe_hist(y, bins)
        m = 0.5 * (p + q)
        kl_pm = np.sum(p * (np.log(p) - np.log(m)))
        kl_qm = np.sum(q * (np.log(q) - np.log(m)))
        return float(0.5 * (kl_pm + kl_qm))

    def causal_divergence(self, treated: np.ndarray, control: np.ndarray) -> float:
        bins = min(32, max(8, int(np.sqrt(min(len(treated), len(control))))))
        return self.js_divergence(treated, control, bins=bins)


class ConvolutionalCausalDynamics:
    def __init__(self, max_memory: int = 20):
        self.max_memory = max_memory
        self.kernels: Dict[Tuple[str, str], ConvolutionalKernel] = {}

    def add_convolution_kernel(
        self, source: str, target: str, kernel: ConvolutionalKernel
    ):
        if kernel._discrete_vals is None:
            kernel.build_discrete(self.max_memory, retention=0.99)
        self.kernels[(source, target)] = kernel

    def convolve_effect(
        self, source_history: np.ndarray, source_var: str, target_var: str
    ) -> float:
        key = (source_var, target_var)
        if key not in self.kernels:
            return 0.0
        kernel = self.kernels[key]
        if kernel._discrete_vals is None:
            kernel.build_discrete(self.max_memory, retention=0.99)
        vals = kernel._discrete_vals
        if vals is None or len(vals) == 0:
            return 0.0
        L = len(vals)
        recent = source_history[-L:]
        edge_gain = float(kernel.parameters.get("edge_gain", 0.1))
        contrib = edge_gain * float(np.sum(recent[::-1] * vals[: len(recent)]))
        return contrib if np.isfinite(contrib) else 0.0

    def propagate_system(
        self,
        current_state: Dict[str, float],
        history: Dict[str, List[float]],
        natural_decay: Dict[str, float],
        dt: float = 1.0,
    ) -> Dict[str, float]:
        new_state = current_state.copy()
        for target_var in current_state.keys():
            total_effect = 0.0
            for source_var in current_state.keys():
                if source_var != target_var and source_var in history:
                    source_history = np.array(history[source_var])
                    total_effect += self.convolve_effect(
                        source_history, source_var, target_var
                    )
            decay = natural_decay.get(target_var, 0.95)
            updated = decay * current_state[target_var] + dt * total_effect
            new_state[target_var] = float(np.clip(updated, -10.0, 10.0))
        return new_state


class TailBoundCalculator:
    @staticmethod
    def empirical_bernstein_bound(
        samples: np.ndarray, confidence: float = 0.95
    ) -> float:
        n = len(samples)
        if n < 2:
            return float("inf")
        sample_var = np.var(samples, ddof=1)
        sample_range = np.max(samples) - np.min(samples)
        delta = 1 - confidence
        log_term = np.log(3 / delta)
        bound = np.sqrt(2 * sample_var * log_term / n) + 3 * sample_range * log_term / n
        return float(bound)


class MCHLErrorAnalysis:
    def __init__(self):
        self.tail_calculator = TailBoundCalculator()

    def compute_estimation_error_bound(
        self, mchl_result: MCHLResult, n_samples: int, confidence: float = 0.95
    ) -> Dict[str, float]:
        bounds = {}
        if mchl_result.fisher_information_matrix is not None:
            try:
                fim_inv = np.linalg.pinv(mchl_result.fisher_information_matrix)
                param_names = [k for k in mchl_result.parameters.keys() if k != "theta"]
                for i, param in enumerate(param_names):
                    if i < fim_inv.shape[0]:
                        se = np.sqrt(max(fim_inv[i, i], 0.0) / max(n_samples, 1))
                        z = norm.ppf((1 + confidence) / 2)
                        bounds[f"{param}_asymptotic"] = float(z * se)
            except Exception:
                pass
        if len(mchl_result.effect_curve) > 1:
            bounds["effect_curve_empirical"] = (
                self.tail_calculator.empirical_bernstein_bound(
                    np.asarray(mchl_result.effect_curve), confidence
                )
            )
        if mchl_result.half_life < float("inf"):
            lam = mchl_result.parameters.get("lambda", 1.0)
            if lam > 0:
                lam_bound = bounds.get("lambda_asymptotic", 0.1)
                hl_derivative = np.log(2) / (lam**2)
                bounds["half_life_delta_method"] = float(abs(hl_derivative) * lam_bound)
        return bounds

    def compute_convergence_rate(self, mchl_result: MCHLResult) -> float:
        standard_rate = 0.5
        if mchl_result.identifiability_conditions:
            ident_score = np.mean(
                [cond.is_satisfied for cond in mchl_result.identifiability_conditions]
            )
            if ident_score < 0.5:
                return standard_rate * 0.7
            elif ident_score < 0.8:
                return standard_rate * 0.85
        return standard_rate


class MCENSystem:
    def __init__(self, variables: List[str], max_time_horizon: int = 50):
        self.variables = variables
        self.max_time_horizon = max_time_horizon
        self.natural_decay = {v: 0.95 for v in variables}
        self.information_metric = PrincipledInformationMetric("jsd")
        self.convolution_dynamics = ConvolutionalCausalDynamics()
        self.error_analysis = MCHLErrorAnalysis()
        self.mchl_results: Dict[str, List[MCHLResult]] = {}
        self.model_ensembles: Dict[str, List[Dict]] = {}
        self.theoretical_guarantees: Dict[str, Dict] = {}
        self._decay_models = [
            NonparametricKernelDecay(bandwidth_grid=None, allow_k0=True),
            ExponentialDecay(),
            PowerLawDecay(),
            BuildupPowerDecay(),
        ]

    def _empirical_kernel_from_effect(
        self, effect_curve: np.ndarray, max_len: int
    ) -> np.ndarray:
        x = np.abs(np.asarray(effect_curve, dtype=float))
        if len(x) >= 3:
            x = np.convolve(x, np.ones(3) / 3.0, mode="same")
        x = x[:max_len] if len(x) >= max_len else x
        if x.sum() <= 0:
            x = np.ones_like(x, dtype=float)
        return x / x.sum()

    def _compute_causal_effects_(
        self,
        data: pd.DataFrame,
        intervention_col: str,
        outcome_col: str,
        time_col: str,
        baseline_intervention: str,
    ) -> Dict[str, Dict]:
        effects = {}
        interventions = data[intervention_col].unique()
        for intervention in interventions:
            if intervention == baseline_intervention:
                continue
            intervention_data = data[data[intervention_col] == intervention]
            baseline_data = data[data[intervention_col] == baseline_intervention]
            time_lags = sorted(
                set(intervention_data[time_col].unique())
                & set(baseline_data[time_col].unique())
            )
            lag_effects, valid_lags, weights = [], [], []
            for lag in time_lags:
                int_outcomes = intervention_data[intervention_data[time_col] == lag][
                    outcome_col
                ].values
                base_outcomes = baseline_data[baseline_data[time_col] == lag][
                    outcome_col
                ].values
                if len(int_outcomes) >= 5 and len(base_outcomes) >= 5:
                    mean_effect = float(np.mean(int_outcomes) - np.mean(base_outcomes))
                    lag_effects.append(mean_effect)
                    valid_lags.append(lag)
                    weights.append(min(len(int_outcomes), len(base_outcomes)))
            if len(lag_effects) >= 3:
                effects[intervention] = {
                    "effects": np.array(lag_effects, dtype=float),
                    "lags": np.array(valid_lags, dtype=float),
                    "n_samples": int(np.mean(weights)) if weights else 1,
                    "weights": np.array(weights, dtype=float) if weights else None,
                }
        return effects

    def _fit_nonparametric_model(
        self,
        model: NonparametricKernelDecay,
        effect_curve: np.ndarray,
        time_lags: np.ndarray,
        weights: Optional[np.ndarray],
    ) -> Tuple[Dict[str, Any], float, np.ndarray, Dict[str, Any]]:
        max_idx = int(np.argmax(np.abs(effect_curve)))
        theta = float(effect_curve[max_idx]) if effect_curve[max_idx] != 0 else 1.0
        norm_val = float(np.abs(theta)) if np.abs(theta) > 1e-8 else 1.0
        y = effect_curve / norm_val
        t = time_lags.astype(float)
        w = np.ones_like(y) if weights is None else (weights / np.max(weights))
        w = np.clip(w, 1e-3, None)
        rng = max(t.max() - t.min(), 1.0)
        step = np.median(np.diff(np.unique(t))) if len(np.unique(t)) > 1 else 1.0
        h_min = max(step * 0.75, rng / 80.0)
        h_max = max(h_min * 1.2, rng / 4.0)
        hs = (
            np.exp(np.linspace(np.log(h_min), np.log(h_max), 10))
            if model.bandwidth_grid is None
            else np.array(model.bandwidth_grid)
        )
        k0_grid = [0.0]
        if model.allow_k0:
            k0_grid = [
                max(0.0, float(p))
                for p in [t.min(), np.percentile(t, 10), np.percentile(t, 20)]
            ]
        best_cv = None
        best_key = None
        for k0, h in product(k0_grid, hs):
            tt = np.clip(t - k0, 0.0, None)
            cv = _kfold_cv_score(tt, y, w, h, k=5)
            if best_cv is None or cv < best_cv:
                best_cv = cv
                best_key = (k0, h)
        k0_star, h_star = best_key
        tt = np.clip(t - k0_star, 0.0, None)
        yhat, lever = _local_linear_smoother(tt, y, tt, h_star, w)
        mu_w = np.average(y, weights=w)
        ss_res = float(np.sum(w * (y - yhat) ** 2))
        ss_tot = float(np.sum(w * (y - mu_w) ** 2))
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
        lo, hi = _moving_block_bootstrap(tt, y, w, tt, h_star, B=200, alpha=0.05)
        params = {
            "bandwidth": float(h_star),
            "k0": float(k0_star),
            "theta": float(norm_val * np.sign(theta)),
            "x_train": tt,
            "y_train": y,
            "w_train": w,
            "bootstrap_lo": lo,
            "bootstrap_hi": hi,
        }
        yhat_scaled = params["theta"] * yhat
        diagnostics = {
            "ss_res": ss_res,
            "n": len(y),
            "k_params": 1,
            "df_eff": float(np.sum(lever)),
            "cv_mse": float(best_cv),
        }
        return params, r2, yhat_scaled, diagnostics

    def _fit__model(
        self,
        model: TheoreticalDecayModel,
        effect_curve: np.ndarray,
        time_lags: np.ndarray,
        weights: Optional[np.ndarray],
    ) -> Tuple[Dict[str, Any], float, np.ndarray, Dict[str, Any]]:
        if isinstance(model, NonparametricKernelDecay):
            return self._fit_nonparametric_model(
                model, effect_curve, time_lags, weights
            )
        max_idx = int(np.argmax(np.abs(effect_curve)))
        theta = float(effect_curve[max_idx]) if effect_curve[max_idx] != 0 else 1.0
        norm_val = float(np.abs(theta)) if np.abs(theta) > 1e-8 else 1.0
        y = effect_curve / norm_val
        t = time_lags.astype(float)
        w = np.ones_like(y) if weights is None else (weights / np.max(weights))
        w = np.clip(w, 1e-3, None)
        if model.name == "exponential":
            initial = {"lambda": 0.1, "k0": float(max(0.0, t[0]))}
            bounds = [(0.001, 10.0), (0.0, float(t.max()))]
        elif model.name == "power_law":
            initial = {"lambda": 1.0, "kappa": 5.0, "k0": float(max(0.0, t[0]))}
            bounds = [(0.1, 5.0), (0.1, 100.0), (0.0, float(t.max()))]
        elif model.name == "buildup_power":
            initial = {
                "lambda_rise": 0.08,
                "lambda_decay": 0.5,
                "kappa": 10.0,
                "k0": float(max(0.0, t[0])),
            }
            bounds = [(0.01, 2.0), (0.1, 5.0), (0.1, 100.0), (0.0, float(t.max()))]
        else:
            initial, bounds = {}, []

        def predict(p):
            k0 = p.get("k0", 0.0)
            tt = np.clip(t - k0, a_min=0.0, a_max=None)
            return model.function(tt, {k: v for k, v in p.items() if k != "k0"})

        def obj(vals):
            pdict = {k: v for k, v in zip(initial.keys(), vals)}
            resid = y - predict(pdict)
            return float(
                np.sum(w * resid**2) + 0.01 * np.sum(np.array(vals, dtype=float) ** 2)
            )

        res = minimize(obj, list(initial.values()), bounds=bounds, method="L-BFGS-B")
        fitted = {
            k: v
            for k, v in zip(
                initial.keys(), (res.x if res.success else list(initial.values()))
            )
        }
        fitted["theta"] = float(norm_val * np.sign(theta))
        yhat = predict(fitted)
        yhat_scaled = fitted["theta"] * yhat
        mu_w = np.average(y, weights=w)
        ss_res = float(np.sum(w * (y - yhat) ** 2))
        ss_tot = float(np.sum(w * (y - mu_w) ** 2))
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
        diagnostics = {
            "ss_res": ss_res,
            "n": len(y),
            "k_params": len([k for k in initial.keys() if k != "k0"]),
        }
        return fitted, r2, yhat_scaled, diagnostics

    def learn__mchl_profiles(
        self,
        interventional_data: pd.DataFrame,
        intervention_col: str = "intervention",
        outcome_col: str = "outcome",
        time_col: str = "time_lag",
        baseline_intervention: str = "control",
    ) -> Dict[str, List[MCHLResult]]:
        results: Dict[str, List[MCHLResult]] = {}
        effects = self._compute_causal_effects_(
            interventional_data,
            intervention_col,
            outcome_col,
            time_col,
            baseline_intervention,
        )
        for intervention, effect_data in effects.items():
            intervention_results: List[MCHLResult] = []
            effect_curve = effect_data["effects"]
            time_lags = effect_data["lags"]
            n_samples = effect_data["n_samples"]
            if len(effect_curve) < 3:
                self.model_ensembles[intervention] = []
                results[intervention] = intervention_results
                continue

            candidates = []
            for model in self._decay_models:
                try:
                    fitted_params, r2_val, yhat_scaled, diag = self._fit__model(
                        model, effect_curve, time_lags, effect_data.get("weights")
                    )
                    tt = np.maximum(0.0, time_lags - fitted_params.get("k0", 0.0))
                    core_params = {
                        k: v
                        for k, v in fitted_params.items()
                        if k not in ("theta", "k0")
                    }
                    identifiability_conditions = model.check_identifiability(
                        tt, core_params
                    )
                    fisher_info = (
                        model.fisher_information(tt, core_params)
                        if hasattr(model, "fisher_information")
                        else None
                    )
                    hl = model.half_life(core_params)
                    mhl_mass = mass_half_life(time_lags, effect_curve)
                    effect_mass = float(np.sum(np.abs(effect_curve)))
                    snr = float(np.sum(effect_curve**2) / (np.var(effect_curve) + 1e-8))

                    n = max(diag.get("n", len(effect_curve)), 3)
                    ss_res = float(diag.get("ss_res", 0.0))
                    sig2 = max(ss_res / n, 1e-10)
                    k_params = int(diag.get("k_params", 1))
                    bic = n * np.log(sig2) + k_params * np.log(n)

                    Tmax = float(np.max(time_lags)) if len(time_lags) else 30.0
                    penalty = 0.2 * max(0.0, (hl - 0.75 * Tmax) / max(1.0, Tmax))
                    bic += penalty

                    candidates.append(
                        {
                            "name": model.name,
                            "params": fitted_params,
                            "r2": r2_val,
                            "yhat": yhat_scaled,
                            "diag": diag,
                            "ident": identifiability_conditions,
                            "fim": fisher_info,
                            "hl": float(hl),
                            "mhl_mass": float(mhl_mass),
                            "effect_mass": float(effect_mass),
                            "snr": float(snr),
                            "bic": float(bic),
                        }
                    )
                except Exception as e:
                    warnings.warn(
                        f"Failed to fit {model.name} for {intervention}: {str(e)}"
                    )
                    continue

            if not candidates:
                self.model_ensembles[intervention] = []
                results[intervention] = intervention_results
                continue

            for c in candidates:
                id_score = np.mean([cond.is_satisfied for cond in c["ident"]])
                c["ident_score"] = id_score
                c["is_identifiable"] = id_score >= 0.8

            identifiable_candidates = [c for c in candidates if c["is_identifiable"]]
            if not identifiable_candidates:
                warnings.warn(
                    f"No identifiable model for {intervention}. Using nonparametric."
                )
                identifiable_candidates = [
                    c for c in candidates if c["name"] == "nonparametric"
                ]

            min_bic = min(c["bic"] for c in identifiable_candidates)
            for c in identifiable_candidates:
                c["posterior_weight"] = np.exp(-0.5 * (c["bic"] - min_bic))
            Z = sum(c["posterior_weight"] for c in identifiable_candidates)
            for c in identifiable_candidates:
                c["posterior_prob"] = c["posterior_weight"] / max(Z, 1e-12)

            identifiable_candidates.sort(
                key=lambda x: x["posterior_prob"], reverse=True
            )
            self.model_ensembles[intervention] = identifiable_candidates

            best = identifiable_candidates[0]
            result = MCHLResult(
                intervention=intervention,
                state_cluster=None,
                decay_type=best["name"],
                parameters=best["params"],
                half_life=best["hl"],
                r2_in_sample=best["r2"],
                effect_curve=effect_curve,
                time_lags=time_lags,
                identifiability_conditions=best["ident"],
                fisher_information_matrix=best["fim"],
                effect_mass=best["effect_mass"],
                snr=best["snr"],
                mass_half_life=best["mhl_mass"],
                posterior_prob=best["posterior_prob"],
            )

            error_bounds = self.error_analysis.compute_estimation_error_bound(
                result, n_samples
            )
            conv_rate = self.error_analysis.compute_convergence_rate(result)
            result.tail_bound = error_bounds.get("effect_curve_empirical", 0.0)
            result.convergence_rate = conv_rate

            intervention_results.append(result)
            ident_score = np.mean(
                [cond.is_satisfied for cond in result.identifiability_conditions]
            )
            print(
                f"{intervention}: {result.decay_type}, "
                f"P(·)={result.posterior_prob:.3f}, "
                f"Ident={ident_score:.2f}, "
                f"HL={result.half_life:.2f}, "
                f"MassHL={result.mass_half_life:.2f}, "
                f"SNR={result.snr:.2f}"
            )

            results[intervention] = intervention_results

        self.mchl_results = results
        return results

    def setup_convolutional_dynamics(
        self, outcome_var: Optional[str] = None, min_r2_for_param: float = 0.35
    ):
        if outcome_var is None:
            outcome_var = self.variables[-1]
        H = min(50, self.convolution_dynamics.max_memory)
        for intervention, results in self.mchl_results.items():
            effect_curve = results[0].effect_curve if results else None
            use_empirical = False
            if results:
                best = results[0]
                id_ok = all(c.is_satisfied for c in best.identifiability_conditions)
                r2_ok = best.r2_in_sample >= min_r2_for_param
                if not id_ok or not r2_ok:
                    use_empirical = True

            if use_empirical and effect_curve is not None:
                vals = self._empirical_kernel_from_effect(effect_curve, H)
                kernel = ConvolutionalKernel(
                    "power_law", len(vals), {"edge_gain": 0.1}, 1.0
                )
                kernel._discrete_vals = vals
                kernel._discrete_lags = np.arange(1, len(vals) + 1)
                if intervention != outcome_var:
                    self.convolution_dynamics.add_convolution_kernel(
                        intervention, outcome_var, kernel
                    )
            elif self.model_ensembles.get(intervention):
                mix = np.zeros(H, dtype=float)
                for c in self.model_ensembles[intervention]:
                    w = float(c.get("posterior_prob", 0.0))
                    if w <= 0:
                        continue
                    lags = np.arange(1, H + 1, dtype=float)
                    k0 = float(c["params"].get("k0", 0.0))
                    tt = np.clip(lags - k0, 0.0, None)
                    if c["name"] == "exponential":
                        g = ExponentialDecay().function(
                            tt, {"lambda": c["params"]["lambda"]}
                        )
                    elif c["name"] == "power_law":
                        g = PowerLawDecay().function(
                            tt,
                            {
                                "lambda": c["params"]["lambda"],
                                "kappa": c["params"]["kappa"],
                            },
                        )
                    elif c["name"] == "buildup_power":
                        g = BuildupPowerDecay().function(
                            tt,
                            {
                                "lambda_rise": c["params"]["lambda_rise"],
                                "lambda_decay": c["params"]["lambda_decay"],
                                "kappa": c["params"]["kappa"],
                            },
                        )
                    elif c["name"] == "nonparametric":
                        g = NonparametricKernelDecay().function(
                            tt,
                            {
                                "bandwidth": c["params"]["bandwidth"],
                                "x_train": c["params"]["x_train"],
                                "y_train": c["params"]["y_train"],
                                "w_train": c["params"].get("w_train"),
                            },
                        )
                    else:
                        continue
                    mix[: len(g)] += w * np.abs(g.astype(float))
                if mix.sum() <= 0 and effect_curve is not None:
                    mix = self._empirical_kernel_from_effect(effect_curve, H)
                else:
                    mix = mix / mix.sum()
                kernel = ConvolutionalKernel(
                    "power_law", len(mix), {"edge_gain": 0.1}, 1.0
                )
                kernel._discrete_vals = mix
                kernel._discrete_lags = np.arange(1, len(mix) + 1)
                if intervention != outcome_var:
                    self.convolution_dynamics.add_convolution_kernel(
                        intervention, outcome_var, kernel
                    )

    def simulate_with_convolution(
        self, intervention_scenario: Dict, time_horizon: int = 20
    ) -> Dict[str, np.ndarray]:
        intervention_var = intervention_scenario["intervention_var"]
        intervention_value = intervention_scenario.get("intervention_value", 1.0)
        initial_state = {
            var: intervention_scenario.get("initial_state", {}).get(var, 0.0)
            for var in self.variables
        }
        initial_state[intervention_var] = intervention_value
        current_state = initial_state.copy()
        history = {var: [current_state[var]] for var in self.variables}
        series = {var: [current_state[var]] for var in self.variables}

        for _ in range(1, time_horizon):
            new_state = self.convolution_dynamics.propagate_system(
                current_state, history, self.natural_decay, dt=1.0
            )
            for var in self.variables:
                history[var].append(new_state[var])
                series[var].append(new_state[var])
                if len(history[var]) > self.convolution_dynamics.max_memory:
                    history[var] = history[var][-self.convolution_dynamics.max_memory :]
            current_state = new_state

        return {var: np.array(series[var]) for var in self.variables}

    def validate_theoretical_properties(self) -> Dict[str, Dict]:
        validation_results = {}
        for intervention, results in self.mchl_results.items():
            intervention_validation = {}
            for i, result in enumerate(results):
                profile_validation = {
                    "identifiability_score": 0.0,
                    "convergence_rate": result.convergence_rate or 0.0,
                    "tail_bound": result.tail_bound or float("inf"),
                    "fisher_information_rank": 0,
                    "parameter_stability": True,
                    "theoretical_guarantees": [],
                }
                if result.identifiability_conditions:
                    satisfied = sum(
                        1 for c in result.identifiability_conditions if c.is_satisfied
                    )
                    total = len(result.identifiability_conditions)
                    profile_validation["identifiability_score"] = satisfied / total
                    for cond in result.identifiability_conditions:
                        profile_validation["theoretical_guarantees"].append(
                            {
                                "condition": cond.condition_name,
                                "satisfied": cond.is_satisfied,
                                "test_statistic": cond.test_statistic,
                                "violation_reason": cond.violation_reason,
                            }
                        )
                if result.fisher_information_matrix is not None:
                    try:
                        rank = int(
                            np.linalg.matrix_rank(result.fisher_information_matrix)
                        )
                        cond_num = float(
                            np.linalg.cond(result.fisher_information_matrix)
                        )
                        profile_validation["fisher_information_rank"] = rank
                        profile_validation["parameter_stability"] = cond_num < 1000
                    except Exception:
                        pass
                if result.convergence_rate:
                    n_effective = 100
                    profile_validation["asymptotic_error_bound"] = float(
                        n_effective ** (-result.convergence_rate)
                    )
                intervention_validation[f"profile_{i}"] = profile_validation
            validation_results[intervention] = intervention_validation
        self.theoretical_guarantees = validation_results
        return validation_results

    def compute_system_wide_information_flow(
        self, max_lag: int = 5
    ) -> Dict[Tuple[str, str], float]:
        flows = {}
        n_samples = 200
        for src in self.variables:
            for tgt in self.variables:
                if src == tgt:
                    continue
                scenario = {
                    "intervention_var": src,
                    "intervention_value": 1.0,
                    "initial_state": {},
                }
                try:
                    ts = self.simulate_with_convolution(
                        scenario, time_horizon=n_samples
                    )
                    x, y = ts[src], ts[tgt]
                    best = 0.0
                    for h in range(1, max_lag + 1):
                        if len(x) > h and len(y) > h:
                            best = max(
                                best,
                                self.information_metric.causal_divergence(
                                    x[h:], y[:-h]
                                ),
                            )
                    flows[(src, tgt)] = float(best)
                except Exception as e:
                    warnings.warn(f"Info flow failed {src}->{tgt}: {e}")
                    flows[(src, tgt)] = 0.0
        return flows

    def export_theoretical_analysis(self, filepath: str):
        analysis_data = {
            "mchl_results": self.mchl_results,
            "theoretical_guarantees": self.theoretical_guarantees,
            "information_flows": self.compute_system_wide_information_flow(),
            "_parameters": {
                "max_time_horizon": self.max_time_horizon,
                "information_metric_type": self.information_metric.metric_type,
                "convolution_max_memory": self.convolution_dynamics.max_memory,
            },
            "summary_statistics": {
                "total_profiles": sum(
                    len(results) for results in self.mchl_results.values()
                ),
                "avg_identifiability_score": np.mean(
                    [
                        v["identifiability_score"]
                        for iv in self.theoretical_guarantees.values()
                        for v in iv.values()
                    ]
                )
                if self.theoretical_guarantees
                else 0.0,
                "avg_convergence_rate": np.mean(
                    [
                        r.convergence_rate
                        for results in self.mchl_results.values()
                        for r in results
                        if r.convergence_rate
                    ]
                )
                if self.mchl_results
                else 0.0,
                "theoretical_guarantees_satisfied": sum(
                    1
                    for iv in self.theoretical_guarantees.values()
                    for v in iv.values()
                    for g in v["theoretical_guarantees"]
                    if g["satisfied"]
                ),
            },
        }
        with open(filepath, "wb") as f:
            pickle.dump(analysis_data, f)
        print(f"Theoretical analysis exported to {filepath}")

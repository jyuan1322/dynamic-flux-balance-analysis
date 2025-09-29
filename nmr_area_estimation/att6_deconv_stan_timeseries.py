import stan
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.optimize import minimize

stan_code = """
functions {
  // Lorentzian peak
  vector lorentzian(vector ppm, real mu, real sigma) {
    return 1 ./ (1 + square((ppm - mu) / sigma));
  }

  // Metabolite mixture
  vector metabolite(vector ppm, vector mu, vector sigma, vector amp) {
    int M = num_elements(mu);
    vector[rows(ppm)] out = rep_vector(0, rows(ppm));
    for (m in 1:M)
      out += amp[m] * lorentzian(ppm, mu[m], sigma[m]);
    return out;
  }
}

data {
  int<lower=1> T;        // number of time points
  int<lower=1> P;        // ppm points
  int<lower=1> M1;       // metabolite 1 peaks
  int<lower=1> M2;       // metabolite 2 peaks
  matrix[T, P] Y;        // observed spectra
  vector[P] ppm;

  vector[M1] mu1;
  vector[M1] sigma1;
  vector[M1] amp1;

  vector[M2] mu2;
  vector[M2] sigma2;
  vector[M2] amp2;
}

parameters {
  vector[T] z;                        // latent weights (0..1)
  real<lower=-1,upper=1> rho;         // AR(1) smoothness
  vector<lower=-0.01, upper=0.01>[T] delta1;  // ppm shifts metabolite 1
  vector<lower=-0.01, upper=0.01>[T] delta2;  // ppm shifts metabolite 2
}

transformed parameters {
  vector<lower=0,upper=1>[T] w;
  matrix[T, P] Y_hat;

  w = inv_logit(z);

  for (t in 1:T) {
    vector[P] met1 = metabolite(ppm + delta1[t], mu1, sigma1, amp1);
    vector[P] met2 = metabolite(ppm + delta2[t], mu2, sigma2, amp2);
    Y_hat[t] = to_row_vector((1 - w[t]) .* met1 + w[t] .* met2);
  }
}

model {
  // Optional AR(1) smoothness penalty
  z[1] ~ normal(0, 1);
  for (t in 2:T)
    z[t] ~ normal(rho * z[t-1], 0.05);  // small sigma acts as smoothing

  rho ~ uniform(-1,1);

  delta1 ~ normal(0, 0.002);
  delta2 ~ normal(0, 0.002);

  // **Euclidean distance objective**
  for (t in 1:T)
    target += -0.5 * dot_self(Y[t] - Y_hat[t]); // equivalent to minimizing sum of squares
}
"""

# Extract Lorentzian params
def extract_params(fit_dict):
    comps = fit_dict["fit_results"]["component_params"]
    mu = [c["center"] for c in comps]
    sigma = [c["sigma"] for c in comps]
    amp = [c["amplitude"] for c in comps]
    return np.array(mu), np.array(sigma), np.array(amp)



def run_nmr_model(Y, ppm, first_fit, last_fit, num_samples=1000, num_chains=4):

mu1, sigma1, amp1 = extract_params(first_fit)
mu2, sigma2, amp2 = extract_params(last_fit)

# Stan data dict
stan_data = {
    "T": Y.shape[0],
    "P": Y.shape[1],
    "M1": len(mu1),
    "M2": len(mu2),
    "Y": Y,
    "ppm": ppm,
    "mu1": mu1,
    "sigma1": sigma1,
    "amp1": amp1,
    "mu2": mu2,
    "sigma2": sigma2,
    "amp2": amp2,
}

# Compile & optimize
posterior = stan.build(stan_code, data=stan_data)
opt_result = posterior.optimizing()

# Extract optimized parameters
w_opt = opt_result["w"]           # shape: (T,)
delta1_opt = opt_result["delta1"] # shape: (T,)
delta2_opt = opt_result["delta2"] # shape: (T,)

time_axis = np.arange(Y.shape[0])

# Plot mixture weight over time
plt.plot(time_axis, w_opt, label="Metabolite 2 fraction")
plt.xlabel("Time index")
plt.ylabel("Fraction metabolite 2")
plt.ylim(0, 1)
plt.legend()
plt.show()





# P = number of ppm points
# met1, met2 are vectors of length P
P = len(ppm)
met1 = np.zeros(P)
met1_comps = []
for comp in first_fit["fit_results"]["component_params"]:
    amp = comp['amplitude']
    mu = comp['center']
    sigma = comp['sigma']
    met1 += amp / (np.pi * sigma) / (1 + ((ppm - mu)/sigma)**2)
    met1_comps.append(amp / (np.pi * sigma) / (1 + ((ppm - mu)/sigma)**2))
  
met1 += first_fit["fit_results"]["baseline_params"]["c"]

# leucine initial deconv
plt.plot(Y[0, :])
plt.plot(met1)
for comp in met1_comps:
    plt.plot(comp)

plt.show()

met2 = np.zeros(P)
met2_comps = []
for comp in last_fit["fit_results"]["component_params"]:
    amp = comp['amplitude']
    mu = comp['center']
    sigma = comp['sigma']
    met2 += amp / (np.pi * sigma) / (1 + ((ppm - mu)/sigma)**2)
    met2_comps.append(amp / (np.pi * sigma) / (1 + ((ppm - mu)/sigma)**2))

met2 += last_fit["fit_results"]["baseline_params"]["c"]

# butyrate initial deconv
plt.plot(Y[-1, :])
plt.plot(met2)
for comp in met2_comps:
    plt.plot(comp)

plt.show()


# met1 = Y[0, :]
# met2 = Y[T-1, :]
def loss(w):
    Y_hat = np.outer(1-w, met1) + np.outer(w, met2)  # T x P
    return np.sum((Y - Y_hat)**2)

# initial guess
T = Y.shape[0]
w0 = np.full(T, 0.5)
res = minimize(loss, w0, bounds=[(0, 1)]*T)
w_opt = res.x

plt.plot(w_opt)
plt.show()








# met1, met2: shape (P,)
# Y: shape (T, P)

lambda_w = 1.0       # adjust to control smoothness
lambda_delta = 1.0   # adjust to control shift smoothness

def loss_pseudo_timeseries(params):
  T = Y.shape[0]
  w = params[:T]                  # first T entries
  delta1 = params[T:2*T]          # next T entries
  delta2 = params[2*T:3*T]        # last T entries
  
  # Apply shifts
  P = len(ppm)
  Y_hat = np.zeros_like(Y)

  interp_met1 = interp1d(ppm, met1, kind="linear", fill_value="extrapolate")
  interp_met2 = interp1d(ppm, met2, kind="linear", fill_value="extrapolate")

  for t in range(T):
    met1_shifted = interp_met1(ppm + delta1[t])
    met2_shifted = interp_met2(ppm + delta2[t])
    Y_hat[t] = (1 - w[t]) * met1_shifted + w[t] * met2_shifted
  
  # Euclidean loss
  mse = np.sum((Y - Y_hat)**2)
  
  # Time-series smoothness penalties
  smooth_w = np.sum(np.diff(w)**2)
  smooth_delta1 = np.sum(np.diff(delta1)**2)
  smooth_delta2 = np.sum(np.diff(delta2)**2)

  return mse + lambda_w * smooth_w + lambda_delta * (smooth_delta1 + smooth_delta2)

T = Y.shape[0]
x0 = np.concatenate([0.5*np.ones(T), np.zeros(T), np.zeros(T)])
bounds = [(0,1)]*T + [(-0.01,0.01)]*T + [(-0.01,0.01)]*T

res = minimize(loss_pseudo_timeseries, x0, bounds=bounds, method='L-BFGS-B')
w_opt = res.x[:T]
delta1_opt = res.x[T:2*T]
delta2_opt = res.x[2*T:3*T]
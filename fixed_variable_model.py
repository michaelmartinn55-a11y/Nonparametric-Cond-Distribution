import yfinance as yf
import numpy as np
import pandas as pd

spy = yf.download("SPY", period="5y", interval="1d")

c = spy["Close"].squeeze()
r = np.log(c).diff()
x = pd.DataFrame({
     "momentum": np.log(c/c.shift(60))
})

array = x.to_numpy()

# ======== Parameters ========
b = 100		#burn-in
h = 4		#condition aggression
f = 5		#forward shift of outcomes
p = 0.005	#kernel bandwidth

# ======== Data Preparation ========
y = np.log(c.shift(-f) / c)
y = y.to_numpy()
mask = (  ~np.isnan(array).any(axis=1) &  ~np.isnan(y) )

array = array[mask]
y = y[mask]

# ======== Density storage ======== 
cond_den = []
uncond_den = []

# ======== Time series loop ========
for t in range (b,len(array)-f):

  hist_array = array[:t]
  current_x = array[t]
  stdev = np.std(hist_array, axis=0)

  valid = (~np.isnan(stdev)) & (stdev != 0)

  hist_array = hist_array[:, valid]
  current_x = current_x[valid]
  stdev = stdev[valid]

  d_n_array = np.sqrt(np.sum( ((hist_array - current_x)/stdev)**2, axis=1 ))
	
  w_i = np.exp(-h * d_n_array**2)

  z = y[t]		# realized outcome for most recent point
  hist_w_i = w_i[:t-f]
  hist_y = y[:t-f]


  k = (1 / np.sqrt(2*np.pi)) * np.exp(-0.5 * ((hist_y - z)/p)**2)

  kde = k * hist_w_i

  c_density_z = np.sum(kde) / np.sum(hist_w_i)

  cond_den.append(c_density_z)

  u_density_z = np.sum(k) / len(hist_y)

  uncond_den.append(u_density_z)


# ======== Log likelihood scores ========

score_c = np.mean(np.log(cond_den))
score_u = np.mean(np.log(uncond_den))


print(f"conditional score:{score_c}")
print(f"unconditional score:{score_u}")

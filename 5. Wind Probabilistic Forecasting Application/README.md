# Tube-Loss-based-Deep-Networks-for-improving-the-probabilistic-forecasting-of-wind-speed
Uncertainty Quantification (UQ) in wind speed
forecasting is a critical challenge in wind power production
due to the inherently volatile nature of wind. By quantify-
ing the associated risks and returns, UQ supports more
effective decision-making for grid operations and partici-
pation in the electricity market. In this paper, we design a
sequence of deep learning based probabilistic forecasting
methods by using the Tube loss function for wind speed
forecasting. The Tube loss function is a simple and model
agnostic Prediction Interval (PI) estimation approach and
can obtain the narrow PI with asymptotical coverage guar-
antees without any distribution assumption. Our deep prob-
abilistic forecasting models effectively incorporate popular
architectures such as LSTM, GRU, and TCN within the
Tube loss framework. We further design a simple yet effec-
tive heuristic for tuning the δ parameter of the Tube loss
function so that our deep forecasting models obtain the
narrower PI without compromising its calibration ability.
We have considered three wind datasets, containing the
hourly recording of the wind speed, collected from three
distinct location namely Jaisalmer, Los Angeles and San
Fransico. Our numerical results demonstrate that the pro-
posed deep forecasting models produce more reliable and
narrower PIs compared to recently developed probabilistic
wind forecasting methods.

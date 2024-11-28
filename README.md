# Improving Prediction Interval Estimation and Deep Probabilistic Forecasting Using the Tube Loss Function


 1. The first loss function that guarantees target t-coverage asymptotically, with proof provided at  
 2. Tube loss can move the PI tube up and down with its r parameter for capturing the denser region of the data cloud for narrow PI Tube.
3.  For symmteric noise distribution, use $r=0.5$. For long-tailed noise distribution, lower or higher values of r provide the narrow PI.
4.  With r =0.5, it is a first contnious loss function for PI estimation and Probablistic Forecasting.
5. Tube loss is easy to backpropagate with the gradient descent method without any approximation.
6. The delta parameter of the Tube loss helps to reduce the width of the PI tube when the obtained PI width is somewhat greater than target t on the validation set. 
7. We have used the Tube loss function for PI estimation in Kernel machines and Artificial Neural Networks and shown that it is far better than its existing alternative.
8. Also, we have used the Tube loss function in GRU, LSTM, and TCN for probabilistic forecasting and obtained very effective forecasts.

   
You can use the Tube loss-based deep forecasting models for probabilistic forecasting of wind power, wave energy, solar irradiance, crypto prices, stock prices, exchange rate, pollution rate,  etc.

# Major-Project
## Accurate Air Quality Index Forecasting using Bi-LSTM Neural Network.
Air quality is a major environmental concern for both society and individuals. Predicting the Air Quality Index (AQI) using machine learning helps analyze future air quality trends. However, using a single machine learning model often struggles to provide accurate predictions due to fluctuations in AQI levels.

To improve prediction accuracy, we enhanced an advanced model called GA-KELM (Genetic Algorithm-based Kernel Extreme Learning Machine). First, we use a kernel method to generate a matrix that improves the model’s learning process. A common issue with traditional models is that random selection of parameters, such as hidden nodes and weights, can reduce accuracy. To solve this, we use a genetic algorithm to optimize these parameters. The model selects the best values by evaluating performance using a fitness function that considers thresholds, weights, and error rates. Finally, we apply the least squares method to determine the output weights.

Genetic algorithms help the model find the best possible solution by continuously improving through an iterative process. To test its effectiveness, we used real air quality data to predict pollutant levels (SO₂, NO₂, PM10, CO, O₃, PM2.5) and AQI. We compared GA-KELM with other models like CMAQ (Community Multiscale Air Quality) and SVR (Support Vector Regression). The results showed that GA-KELM trains faster and makes more accurate predictions.

Additionally, we explored the Bi-LSTM (Bidirectional Long Short-Term Memory) algorithm, which enhances predictions by analyzing data in both forward and backward directions, further optimizing feature weights for better accuracy.

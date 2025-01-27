# Air Quality Forecasting Using LSTM

This project aims to predict the concentration of PM2.5 (particulate matter) in Beijing using historical air quality and weather data. The primary objective is to predict future PM2.5 levels, enabling timely actions to mitigate the effects of air pollution. The model is built using Long Short-Term Memory (LSTM) networks, which are particularly well-suited for time-series forecasting tasks.

## Project Overview

### Problem Statement

Air pollution, particularly PM2.5 concentrations, is a significant public health issue worldwide. In this project, I applied Recurrent Neural Networks (RNNs) and LSTM models to forecast PM2.5 levels in Beijing using historical data. The ability to forecast air quality levels accurately can help authorities make informed decisions to protect public health.

### Dataset

The dataset used for this project consists of historical air quality data from Beijing, including PM2.5 concentration, temperature, humidity, wind speed, and other weather parameters. The dataset is split into training and test sets, with the goal of using the training data to train the model and the test data to evaluate its performance.

- **train.csv**: Training data containing features like temperature, humidity, wind speed, and PM2.5 levels.
- **test.csv**: Test data for evaluating the model's performance.
- **sample_submission.csv**: A sample file showing the required format for submitting predictions.

### Objective

The objective of this project is to forecast the PM2.5 concentration for future time steps using the historical data. The model will be trained on the training dataset and tested on the test dataset. The predictions will then be submitted for evaluation in a Kaggle competition.

## How It Works

### Data Preprocessing

The data preprocessing steps involved:
- **Handling Missing Values**: Missing data was filled using the mean of each feature.
- **Feature Scaling**: The features were scaled to a range of 0 to 1 using Min-Max scaling to improve the performance of the neural network.
- **Sequence Generation**: The time-series data was transformed into sequences to capture temporal dependencies. A window size of 24 hours was used to generate input-output pairs for the model.

### Model Architecture

The model architecture consists of:
- **Input Layer**: Data representing air quality and weather features.
- **LSTM Layers**: Two LSTM layers were used, with 64 units each, to capture long-term dependencies in the time-series data.
- **Dropout Layer**: A dropout layer was added to prevent overfitting.
- **Output Layer**: A single neuron to predict the PM2.5 concentration for the next time step.

The model was trained using the **Adam optimizer** and the **Mean Squared Error (MSE)** loss function for 100 epochs with a batch size of 64.

### Experiments

I conducted five different experiments with varying parameters (learning rate and number of layers) to find the best-performing model. The table below summarizes the experiments:

| Experiment # | Parameters                                  | Model Architecture                                | Output (RMSE) |
|--------------|---------------------------------------------|--------------------------------------------------|---------------|
| 1            | Learning rate = 0.01, Batch size = 32       | Single LSTM layer, 128 units                    | 35.45         |
| 2            | Learning rate = 0.001, Batch size = 64      | Two LSTM layers, 64 units                       | 29.87         |
| 3            | Learning rate = 0.001, Batch size = 32      | Two LSTM layers, 128 units                      | 32.58         |
| 4            | Learning rate = 0.0005, Batch size = 64     | Single LSTM layer, 64 units                     | 30.72         |
| 5            | Learning rate = 0.0001, Batch size = 128    | Two LSTM layers, 256 units                      | 28.64         |

The model with two LSTM layers (64 units) and a learning rate of 0.001 gave the best performance with an RMSE of 29.87.

## Requirements

To run this project, you will need the following libraries:

- Python 3.x
- TensorFlow 2.x
- Pandas
- NumPy
- Matplotlib
- Scikit-learn

You can install the required libraries using `pip`:

```bash
pip install -r requirements.txt
requirements.txt
makefile
Copy
Edit
tensorflow==2.7.0
pandas==1.3.3
numpy==1.21.2
matplotlib==3.4.3
scikit-learn==0.24.2
Running the Code
Clone the repository:

bash
Copy
Edit
git clone https://github.com/yourusername/air-quality-forecasting.git
cd air-quality-forecasting
Prepare the dataset by placing the train.csv, test.csv, and sample_submission.csv files in the root directory.

Run the Jupyter notebook or Python script to train the model and generate predictions.

bash
Copy
Edit
jupyter notebook air_quality_forecasting.ipynb
Or run the script directly if you prefer:

bash
Copy
Edit
python air_quality_forecasting.py
After training, the model will generate a subm_fixed.csv file containing the predictions for submission.

Submit the subm_fixed.csv file on Kaggle for evaluation.

Conclusion
This project demonstrated the use of LSTM networks for forecasting air quality (PM2.5) concentrations. By preprocessing the data, training the LSTM model, and evaluating its performance, I successfully generated predictions that can be submitted to Kaggle for further evaluation. The results from the experiments showed that the model with two LSTM layers and a learning rate of 0.001 provided the best performance.

Future Work
Explore additional features such as weather forecasts to improve prediction accuracy.
Try advanced techniques like attention mechanisms to capture long-term dependencies better.
Use hyperparameter optimization (e.g., grid search) to fine-tune the model further.
GitHub Repository
You can view and clone the GitHub repository for this project at [GitHub Link].

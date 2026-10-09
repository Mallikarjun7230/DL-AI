# Numerical Regression Model — Step-by-Step Training

## Project Overview
This project builds a machine learning model that learns a relationship from generated numerical data and predicts outputs for unseen inputs.

The generated data follows approximately:

`y = 3x + 2 + noise`

A linear regression model is trained using gradient descent. The notebook includes data generation, train-test splitting, model training, a loss curve, predictions, evaluation metrics, and an actual-versus-predicted graph.

## Project Files
- `numerical_regression_training.ipynb` — complete step-by-step Jupyter Notebook.
- `actual_vs_predicted.png` — graph comparing actual and predicted test values.
- `loss_curve.png` — graph showing training loss across epochs.
- `README.md` — project documentation.

## Requirements
- Python 3
- NumPy
- Matplotlib
- Jupyter Notebook or Google Colab

Install the required libraries:

```bash
pip install numpy matplotlib
```

## How to Run
1. Download the project files.
2. Open `numerical_regression_training.ipynb` in Jupyter Notebook, JupyterLab, or Google Colab.
3. Install the requirements if they are not already installed.
4. Run the notebook cells from top to bottom.
5. Review the printed predictions, evaluation metrics, and generated graphs.

## Training Process
1. **Import libraries:** Load NumPy and Matplotlib.
2. **Generate data:** Create 200 input-output samples with a linear relationship and small random noise.
3. **Split data:** Use 160 samples for training and 40 samples for testing.
4. **Initialize parameters:** Start the model weight (`w`) and bias (`b`) at zero.
5. **Train the model:** Run 300 epochs of gradient descent, calculating Mean Squared Error (MSE) and updating the parameters.
6. **Plot training loss:** Visualize how the training MSE changes over epochs.
7. **Predict unseen values:** Use the trained model on the test inputs.
8. **Evaluate performance:** Calculate MSE, Mean Absolute Error (MAE), and R-squared (R²).
9. **Plot predictions:** Compare actual test outputs with predicted outputs.

## Evaluation Metrics
- **MSE (Mean Squared Error):** Average squared difference between actual and predicted values. Lower is better.
- **MAE (Mean Absolute Error):** Average absolute difference between actual and predicted values. Lower is better.
- **R² (Coefficient of Determination):** Indicates how much variation in the target values is explained by the model. A value closer to 1 generally indicates a better fit.

## Results
For the included reproducible run, the model learned parameters close to the data-generating relationship `y = 3x + 2`.

| Metric | Result |
|---|---:|
| Training samples | 160 |
| Testing samples | 40 |
| Epochs | 300 |
| Learned weight | 3.058 |
| Learned bias | 2.019 |
| Test MSE | 0.0411 |
| Test MAE | 0.1598 |
| Test R² | 0.9861 |

Exact results may vary if the data-generation process, random seed, or training settings are changed.

## Graphs
### Training Loss Curve
The loss curve should generally decrease as gradient descent learns better parameter values. A curve that levels off suggests training has stabilized.

![Training Loss Curve](loss_curve.png)

### Actual vs. Predicted
Points close to the diagonal line indicate predictions close to the actual values.

![Actual vs Predicted](actual_vs_predicted.png)

## Conclusion
This project demonstrates a basic machine learning workflow: generating numerical data, splitting it into training and testing sets, training a model with gradient descent, monitoring loss, predicting unseen inputs, and evaluating the results.

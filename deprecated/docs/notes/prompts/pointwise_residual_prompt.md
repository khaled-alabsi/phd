Train a pointwise residual regressor pipeline on Tennessee Eastman data.

Steps:
1. Split scaled sensor data by simulation run into train/val/test.
2. Fit the autoencoder teacher on train data (early stopping on val); save weights/config.
3. Compute AE reconstruction residuals for train/val; use them as regression targets.
4. Train pointwise DNN to predict residuals from current sensor vector.
   - Support optional hyperparameter grid search; track best MAE.
5. Evaluate on val/test: log MAE, RMSE, R² vs AE residuals; visualise if needed.
6. Cache artefacts (AE weights, DNN weights, metrics, best config) unless `RESET_EXPERIMENT=True`.
7. Expose inference helpers: predict residuals, compute AE residuals, load cached models.

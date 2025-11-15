# Public Benchmark Datasets
Standard reference datasets help compare multivariate charts. Examples used in literature include:
- UCI SECOM dataset: Semiconductor manufacturing data (1567 samples, 591 sensor readings)​
archive.ics.uci.edu
. Used for anomaly and quality detection in many studies.
Kaggle “Predicting Manufacturing Defects” dataset: Synthetic multivariate manufacturing metrics (17 features, classification task)​
github.com
Contains supply-chain, production, and quality variables; often used for defect-prediction modeling.

- Kaggle Multistage Continuous-Flow Manufacturing (MCMP): Data from a continuous production line (~14,000 records)​
medium.com
. Includes process measurements across stages; used in recent studies for fault detection.

- Tennessee Eastman Process (simulation): A widely used chemical process simulation (41 variables) for fault detection benchmarking (though not a public “dataset” per se).

- Others: Some SPC studies use datasets from NIST, automotive production, or aerospace (via proprietary or public sources). These chart data tend to be high-dimensional and correlated, making them suitable for comparing T², EWMA, and CUSUM methods.
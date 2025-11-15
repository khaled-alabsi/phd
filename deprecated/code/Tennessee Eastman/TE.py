# Converting .rData into dataframe
fault_free_training_dict = pyreadr.read_r("data/TEP_FaultFree_Training.RData")
fault_free_testing_dict = pyreadr.read_r("data/TEP_FaultFree_Testing.RData")

faulty_training_dict = pyreadr.read_r("data/TEP_Faulty_Training.RData")
faulty_testing_dict = pyreadr.read_r("data/TEP_Faulty_Testing.RData")

DF_FF_TRAINING_RAW = fault_free_training_dict["fault_free_training"]
DF_FF_TEST_RAW = fault_free_testing_dict["fault_free_testing"]

DF_F_TRAINING_RAW = faulty_training_dict["faulty_training"]
DF_F_TEST_RAW = faulty_testing_dict["faulty_testing"]

TARGET_VARIABLE_COLUMN_NAME = "faultNumber"
SIMULATION_RUN_COLUMN_NAME = "simulationRun"
COLUMNS_TO_REMOVE = ["simulationRun", "sample"]
SKIPED_FAULTS = []å
FAULTS_TO_BE_MERGED_TOGETHER = [3, 8,9,18, 15]
MERGE_FAUTS_TO_NUMBER = 3
FAULT_INJECTION_STARTING_POINT = 25

DF_F_TRAIN_SKIPPED_FAULTS = DF_F_TRAINING_RAW[~DF_F_TRAINING_RAW[TARGET_VARIABLE_COLUMN_NAME].isin(SKIPED_FAULTS)].reset_index(drop=True)
DF_F_TEST_SKIPPED_FAULTS = DF_F_TEST_RAW[~DF_F_TEST_RAW[TARGET_VARIABLE_COLUMN_NAME].isin(SKIPED_FAULTS)].reset_index(drop=True)

# **Reduce Training and test data for simplicity during development and testing**
# **** THE DATA SHOULD STAY BALANCED !!!! *****

# reduce training data
DF_FF_TRAINING_REDUCED = DF_FF_TRAINING_RAW[(DF_FF_TRAINING_RAW[SIMULATION_RUN_COLUMN_NAME] > 0) & (DF_FF_TRAINING_RAW[SIMULATION_RUN_COLUMN_NAME] < 2)].drop(columns=COLUMNS_TO_REMOVE, axis=1)
DF_F_TRAINING_REDUCED = DF_F_TRAIN_SKIPPED_FAULTS[(DF_F_TRAIN_SKIPPED_FAULTS[SIMULATION_RUN_COLUMN_NAME] > 4 )& (DF_F_TRAIN_SKIPPED_FAULTS[SIMULATION_RUN_COLUMN_NAME] < 6) &(DF_F_TRAIN_SKIPPED_FAULTS["sample"] > FAULT_INJECTION_STARTING_POINT)].drop(columns=COLUMNS_TO_REMOVE, axis=1)

# reduce test data
DF_FF_TEST_REDUCED = DF_FF_TEST_RAW[(DF_FF_TEST_RAW[SIMULATION_RUN_COLUMN_NAME] > 2) & (DF_FF_TEST_RAW[SIMULATION_RUN_COLUMN_NAME] < 4)].drop(columns=COLUMNS_TO_REMOVE, axis=1)
DF_F_TEST_REDUCED = DF_F_TEST_SKIPPED_FAULTS[(DF_F_TEST_SKIPPED_FAULTS[SIMULATION_RUN_COLUMN_NAME] > 5)& (DF_F_TEST_SKIPPED_FAULTS[SIMULATION_RUN_COLUMN_NAME] < 7) &(DF_F_TEST_SKIPPED_FAULTS["sample"] > FAULT_INJECTION_STARTING_POINT)].drop(columns=COLUMNS_TO_REMOVE, axis=1)

# Prepare data for Supervised training and testing

DF_TRAINING_REDUCED_CONCATED = pd.concat([DF_FF_TRAINING_REDUCED, DF_F_TRAINING_REDUCED])
DF_TEST_REDUCED_CONCATED = pd.concat([DF_FF_TEST_REDUCED, DF_F_TEST_REDUCED])


# Standardize the data: It centers the data around 0 and scales it based on standard deviation.
sc = StandardScaler()
sc.fit(DF_TRAINING_REDUCED_CONCATED.drop(columns=[TARGET_VARIABLE_COLUMN_NAME],axis=1))
X_TRAIN = sc.transform(DF_TRAINING_REDUCED_CONCATED.drop(columns=[TARGET_VARIABLE_COLUMN_NAME],axis=1))
Y_TRAIN_DF = DF_TRAINING_REDUCED_CONCATED[TARGET_VARIABLE_COLUMN_NAME]

# Encode the target variable: -LabelEncoder() Takes a list/array of categorical labels (e.g., ['shift', 'trend', 'none']), -Assigns a unique integer to each category (e.g., ['none' → 0, 'shift' → 1, 'trend' → 2]), -Returns a NumPy array of integers corresponding to the original labels
le = LabelEncoder()
Y_TRAIN = le.fit_transform(Y_TRAIN_DF)

# Set the features and target variable for testing
X_TEST_REDUCED = sc.transform(DF_TEST_REDUCED_CONCATED.drop(columns=[TARGET_VARIABLE_COLUMN_NAME], axis=1))
Y_TEST_REDUCED_DF = DF_TEST_REDUCED_CONCATED[TARGET_VARIABLE_COLUMN_NAME]

# Encode the target variable for testing
Y_TEST_REDUCED = le.fit_transform(Y_TEST_REDUCED_DF)

# One-hot encode the target variable
encoder_1 = OneHotEncoder(sparse_output=False)
Y_reshabed = (DF_TRAINING_REDUCED_CONCATED[TARGET_VARIABLE_COLUMN_NAME].to_numpy().reshape(-1, 1))
# .fit_transform() fits the encoder to the data and then transforms it.
# Use this on the training data to learn the encoding mapping and apply it.
Y_ENC_TRAIN = encoder_1.fit_transform(Y_reshabed)

# .transform() only applies the learned encoding to new data.
# Use this on test data to encode using the mapping learned from training data.
Y_test_reshabed = (DF_TEST_REDUCED_CONCATED[TARGET_VARIABLE_COLUMN_NAME].to_numpy().reshape(-1, 1))
Y_ENC_TEST_REDUCED = encoder_1.transform(Y_test_reshabed)

print(
    "Training Data with the following Fault Numbers:",
    DF_TEST_REDUCED_CONCATED[TARGET_VARIABLE_COLUMN_NAME].unique(),
)
DF_TRAINING_REDUCED_CONCATED.head()


# Standardize the data Unsupervised learning

# TRAIN data

# X
X_INCONTROL_TRAIN_REDUCED_DF = DF_FF_TRAINING_REDUCED.drop(columns=[TARGET_VARIABLE_COLUMN_NAME], axis=1)
sc = StandardScaler()
sc.fit(X_INCONTROL_TRAIN_REDUCED_DF)
X_INCONTROL_TRAIN_REDUCED = sc.transform(X_INCONTROL_TRAIN_REDUCED_DF)

# Y
Y_TRAIN_ANOMALY_REDUCED_DF = DF_TRAINING_REDUCED_CONCATED[TARGET_VARIABLE_COLUMN_NAME].apply(lambda x: 0 if x == 0 else 1) ## change target variable to only 2 classes: 0 and 1: 0 is in control , bigger than 0 is faulty
encoder_2 = OneHotEncoder(sparse_output=False)
Y_reshabed = Y_TRAIN_ANOMALY_REDUCED_DF.to_numpy().reshape(-1, 1)
Y_ENC_ANOMALY_TRAIN_REDUCED = encoder_2.fit_transform(Y_reshabed)



# Test data

# X
X_INCONTROL_TEST_REDUCED_DF = DF_FF_TEST_REDUCED.drop( columns=[TARGET_VARIABLE_COLUMN_NAME], axis=1)
sc = StandardScaler()
sc.fit(X_INCONTROL_TEST_REDUCED_DF)
X_INCONTROL_TEST_REDUCED = sc.transform(X_INCONTROL_TEST_REDUCED_DF)

X_OUT_OF_CONTROL_TEST_REDUCED_DF = DF_F_TEST_REDUCED.drop(columns=[TARGET_VARIABLE_COLUMN_NAME], axis=1)
sc = StandardScaler()
sc.fit(X_OUT_OF_CONTROL_TEST_REDUCED_DF)
X_OUT_OF_CONTROL_TEST_REDUCED = sc.transform(X_OUT_OF_CONTROL_TEST_REDUCED_DF)

# Y
Y_TEST_ANOMALY_REDUCED_DF = DF_TEST_REDUCED_CONCATED[TARGET_VARIABLE_COLUMN_NAME].apply(lambda x: 0 if x == 0 else 1) ## change target variable to only 2 classes: 0 and 1: 0 is in control , bigger than 0 is faulty
Y_test_reshabed = Y_TEST_ANOMALY_REDUCED_DF.to_numpy().reshape(-1, 1)
Y_ENC_ANOMALY_TEST_REDUCED = encoder_2.transform(Y_test_reshabed)

y_test_binary = Y_test_reshabed.ravel().tolist()



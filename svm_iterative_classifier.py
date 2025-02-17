# pip install pandas numpy scikit-learn
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.svm import SVC
from pandas import DataFrame as df
import json

def preprocess_dataset(file_path, class_column='class'):
    # Load dataset
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        raise ValueError(f"File '{file_path}' not found.")
    
    # Normalize all column names to lowercase
    df.columns = df.columns.str.lower()  # Convert all column names to lowercase
    class_column = class_column.lower()  # Ensure the provided column name is also lowercase
    
    # Check if class column exists after normalization
    if class_column not in df.columns:
        raise ValueError(f"Column '{class_column}' not found in the dataset. Available columns: {df.columns.tolist()}")
    
    # Drop whole rows with missing values
    df = df.dropna(how='all')

    # Encode class labels if necessary
    label_encoder = None
    original_classes = None
    if df[class_column].dtype == 'object':
        label_encoder = LabelEncoder()
        original_classes = df[class_column].unique()
        df[class_column] = label_encoder.fit_transform(df[class_column])
    
    # Extract features and labels
    X = df.drop(columns=[class_column]).values
    y = df[class_column].values
    
    # Store original class mapping
    class_mapping = {i: label for i, label in enumerate(original_classes if original_classes is not None else np.unique(y))}
    
    # Normalize feature values using Min-Max Scaling
    scaler = MinMaxScaler()
    X_normalized = scaler.fit_transform(X)
    
    return X_normalized, y, class_mapping, label_encoder, df

def iterative_svm(C, X_normalized, y, class_mapping, label_encoder, df, pure_threshold=1.0, save_overlap=True, class_column='class'):
    # Initialize iteration tracking
    iteration = 0
    results_list = []
    
    # Keep track of remaining indices
    remaining_indices = np.arange(len(X_normalized))
    
    while True:
        svm_model = SVC(kernel='linear', C=C)
        svm_model.fit(X_normalized[remaining_indices], y[remaining_indices])
        
        # Compute decision function values for remaining data
        decision_values = svm_model.decision_function(X_normalized[remaining_indices])
        predictions = svm_model.predict(X_normalized[remaining_indices])
        
        # Identify first misclassified cases at each end
        misclassified = np.where(predictions != y[remaining_indices])[0]
        if len(misclassified) > 0:
            lower_bound = decision_values[misclassified].min()
            upper_bound = decision_values[misclassified].max()
        else:
            lower_bound, upper_bound = -pure_threshold, pure_threshold  # Default if no misclassifications
        
        # Identify pure and overlap cases within remaining data
        pure_above_mask = decision_values > upper_bound
        pure_below_mask = decision_values < lower_bound
        overlap_mask = ~(pure_above_mask | pure_below_mask)
        
        # Get indices relative to original dataset
        pure_above = remaining_indices[pure_above_mask]
        pure_below = remaining_indices[pure_below_mask]
        overlap_region = remaining_indices[overlap_mask]
        
        # Extract SVM function
        coef = svm_model.coef_[0]
        intercept = svm_model.intercept_[0]
        svm_function = f"Decision Function: {coef} * X + {intercept}"
        
        # Save overlap cases to CSV if enabled
        if save_overlap and len(overlap_region) > 0:
            overlap_df = pd.DataFrame(X_normalized[overlap_region], columns=df.drop(columns=[class_column]).columns)
            overlap_df['Class'] = y[overlap_region]
            if label_encoder is not None:
                overlap_df['Class'] = label_encoder.inverse_transform(overlap_df['Class'])
            overlap_df.to_csv(f"overlap_cases_iter_{iteration}.csv", index=False)
        
        # Store iteration results with consistent class mapping
        overlap_class_counts = {}
        if len(overlap_region) > 0:
            for class_idx in class_mapping:
                class_name = class_mapping[class_idx]
                overlap_class_counts[class_name] = sum(y[overlap_region] == class_idx)
                
        results = {
            "Iteration": iteration,
            "Pure Cases Above Threshold": len(pure_above),
            "Pure Cases Below Threshold": len(pure_below),
            "Overlap Region Cases": len(overlap_region),
            "Overlap Classes": overlap_class_counts,
            "SVM Function": svm_function,
            "Pure Region Thresholds": {"Lower Bound": lower_bound, "Upper Bound": upper_bound},
            "Overlap Cases Saved": len(overlap_region) > 0
        }
        results_list.append(results)
        
        # Stop if there are no overlap cases left or only one class remains
        if len(overlap_region) == 0 or len(np.unique(y[overlap_region])) == 1:
            break
        
        # Update remaining indices to only include overlap cases
        remaining_indices = overlap_region
        iteration += 1
    
    return iteration, results_list

def find_best_C(X_normalized, y, C_values=[0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000, 100000, 1000000], max_iterations=100):
    best_C = None
    best_accuracy = 0
    for C in C_values:
        svm_model = SVC(kernel='linear', C=C)
        svm_model.fit(X_normalized, y)
        predictions = svm_model.predict(X_normalized)
        accuracy = np.mean(predictions == y)
        if accuracy > best_accuracy:
            best_C = C
            best_accuracy = accuracy
    return best_C

def preprocess_and_run_svm(file_path, class_column='class', pure_threshold=1.0, save_overlap=False):
    # Set seed
    np.random.seed(42)
    
    # Preprocess dataset
    try:
        X_normalized, y, class_mapping, label_encoder, df = preprocess_dataset(file_path, class_column)
    except ValueError as e:
        print(f"Error processing file: {e}")
        return 0, []
    
    # Find the best C value for SVM
    best_C = find_best_C(X_normalized, y)
    print(f"Best C value: {best_C}")
    # Run SVM with the best C value
    iterations, results = iterative_svm(best_C, X_normalized, y, class_mapping, label_encoder, df, pure_threshold, save_overlap, class_column)
    
    return iterations, results

def convert_dict(d):
    """ Recursively convert all dictionary keys and values to JSON-serializable types. """
    new_dict = {}
    for key, value in d.items():
        # Convert NumPy integer keys to Python int
        if isinstance(key, (np.int64, np.int32, np.integer)):
            key = int(key)  
        elif isinstance(key, (np.float64, np.float32, np.floating)):  
            key = float(key)  # Convert NumPy float keys to Python float
        else:
            key = str(key)  # Convert anything else to string (JSON-safe)

        # Convert values
        if isinstance(value, (np.int64, np.int32, np.integer)):
            value = int(value)
        elif isinstance(value, (np.float64, np.float32, np.floating)):
            value = float(value)
        elif isinstance(value, dict):
            value = convert_dict(value)  # Recurse if the value is a nested dictionary
        new_dict[key] = value
    return new_dict

# Run the SVM pipeline
iterations, results = preprocess_and_run_svm("mnist_2_7.csv")
print(f"Total iterations: {iterations}")

# Process results safely
for res in results:
    safe_res = convert_dict(res)  # Convert all keys and values
    print(json.dumps(safe_res, indent=4))
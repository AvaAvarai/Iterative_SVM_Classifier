import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from sklearn.datasets import make_classification

def load_and_normalize_csv(file_path):
    """Load a CSV file, normalize numerical data to [0,1], and extract labels."""
    # Load dataset
    df = pd.read_csv(file_path)

    # Identify the class column (case-insensitive)
    class_col = [col for col in df.columns if col.lower() == "class"]
    if not class_col:
        raise ValueError("No 'class' column found in dataset.")
    class_col = class_col[0]  # Get actual column name

    # Extract labels and numerical features
    labels = df[class_col].values
    feature_cols = [col for col in df.columns if col != class_col]
    data = df[feature_cols].values

    # Min-Max normalize to [0,1]
    scaler = MinMaxScaler()
    normalized_data = scaler.fit_transform(data)

    return normalized_data, labels, feature_cols

def train_svm(X, y, class_label, C=1.0):
    """Train a linear SVM for a given class vs. all other classes."""
    y_binary = np.where(y == class_label, 1, -1)  # One-vs-All labeling
    if len(np.unique(y_binary)) < 2:  # Stop if only one class remains
        return None
    model = SVC(kernel='linear', C=C)
    model.fit(X, y_binary)
    return model

def extract_pure_regions(X, y, model, class_label, min_size_threshold=0.05):
    """Extract high-confidence pure regions using decision function values."""
    if model is None:
        return None, np.ones_like(y, dtype=bool), None, 0  # No hyperplane available
    
    decision_values = model.decision_function(X)
    pure_mask = (decision_values > 0) # TODO: adjust this threshold.

    case_count = np.sum(pure_mask)

    if case_count / len(y) < min_size_threshold or case_count == 0:
        return None, np.ones_like(y, dtype=bool), None, 0  # No valid pure region found

    pure_X = X[pure_mask]
    pure_y = y[pure_mask]
    remaining_mask = ~pure_mask  # Remove extracted pure cases

    return (pure_X, pure_y), remaining_mask, (model.coef_, model.intercept_), case_count

def iterative_svm_pure_extraction(X, y, class_labels, max_iterations=50, min_size_threshold=0.05):
    """Iteratively extract pure regions using multiple SVM hyperplanes per iteration until no cases remain unaccounted for."""
    remaining_X, remaining_y = X.copy(), y.copy()
    pure_regions_all_iterations = {label: [] for label in class_labels}
    hyperplanes_all_iterations = []

    for iteration in range(1, max_iterations + 1):
        if len(np.unique(remaining_y)) == 1:  # Stop if only one class remains
            print(f"\nOnly one class ({np.unique(remaining_y)[0]}) remains. Stopping iteration.")
            break

        new_pure_regions = False  # Track if pure regions are found in this iteration
        iteration_hyperplanes = []
        cases_removed = 0

        print(f"\nIteration {iteration}:")
        
        for class_label in class_labels:
            if len(remaining_y) == 0:  # Stop if dataset is empty
                break
            
            model = train_svm(remaining_X, remaining_y, class_label)
            pure_region, mask, hyperplane, case_count = extract_pure_regions(remaining_X, remaining_y, model, class_label, min_size_threshold)

            if pure_region:
                pure_regions_all_iterations[class_label].append(pure_region)
                remaining_X, remaining_y = remaining_X[mask], remaining_y[mask]  # Update remaining data
                iteration_hyperplanes.append((class_label, hyperplane, case_count))
                cases_removed += case_count
                new_pure_regions = True  # Found at least one pure region
                print(f"  - Class {class_label}: {case_count} cases extracted.")

        hyperplanes_all_iterations.append(iteration_hyperplanes)

        print(f"  Cases removed this iteration: {cases_removed}")
        print(f"  Remaining overlap cases: {len(remaining_y)}")

        if not new_pure_regions:  # Stop if no more pure regions can be extracted
            break

    return pure_regions_all_iterations, remaining_X, remaining_y, hyperplanes_all_iterations

# Example usage
csv_file = "breast-cancer-wisconsin-diagnostic.csv"  # Replace with actual CSV path
X, y, feature_names = load_and_normalize_csv(csv_file)

# Output data summary
print(f"Loaded {X.shape[0]} samples with {X.shape[1]} features.")
print(f"Feature names: {feature_names}")
print(f"Label sample: {np.unique(y)}")

# Run the pure region extraction algorithm
class_labels = np.unique(y)
pure_regions, overlap_X, overlap_y, extracted_hyperplanes = iterative_svm_pure_extraction(X, y, class_labels)

# Output results
for class_label in class_labels:
    print(f"\nClass {class_label}: {len(pure_regions[class_label])} pure regions extracted.")

print(f"\nFinal Overlap region size: {len(overlap_X)}")

# Print extracted hyperplanes per iteration
for i, hyperplanes in enumerate(extracted_hyperplanes):
    print(f"\nIteration {i+1}: Extracted Hyperplanes")
    for class_label, (coef, intercept), case_count in hyperplanes:
        print(f"  - Class {class_label}: {case_count} cases extracted, Coefficients: {coef}, Intercept: {intercept}")

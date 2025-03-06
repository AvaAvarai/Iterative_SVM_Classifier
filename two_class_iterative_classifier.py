# Install required packages with:
# pip install numpy pandas matplotlib scikit-learn

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.preprocessing import MinMaxScaler
from tkinter import Tk, filedialog

# Class for tree nodes
class TreeNode:
    def __init__(self, iteration, removed_class, removed_points):
        self.iteration = iteration
        self.removed_class = removed_class
        self.removed_points = removed_points
        self.left = None  # Left branch (one class removed)
        self.right = None  # Right branch (other class removed)
        self.decision_function = None  # Store SVM decision function

# Function to load and preprocess dataset
def load_dataset():
    Tk().withdraw()  # Hide the Tkinter GUI
    file_path = filedialog.askopenfilename(title="Select CSV file", filetypes=[("CSV files", "*.csv")])
    if not file_path:
        raise ValueError("No file selected.")
    
    df = pd.read_csv(file_path)
    
    # Ensure 'class' column exists (case-insensitive search)
    class_col = [col for col in df.columns if col.lower() == "class"]
    if not class_col:
        raise ValueError("Dataset must have a column named 'class'.")
    class_col = class_col[0]  # Use the first matched column
    
    # Separate features and labels
    X = df.drop(columns=[class_col]).values
    y = df[class_col].values
    
    # Normalize features
    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)
    
    # Check if there are more than 2 classes
    unique_classes = np.unique(y)
    if len(unique_classes) > 2:
        raise ValueError("Dataset must have exactly 2 classes.")
    
    return X, y

# Iterative SVM refinement algorithm
def iterative_svm_refinement(X, y, max_iterations=10, kernel='linear'):
    remaining_X, remaining_y = X.copy(), y.copy()
    root = TreeNode(iteration=0, removed_class=None, removed_points=0)
    current_node = root
    iteration_count = 0
    
    while iteration_count < max_iterations and len(np.unique(remaining_y)) > 1:
        iteration_count += 1
        
        # Train SVM
        svm = SVC(kernel=kernel)
        svm.fit(remaining_X, remaining_y)
        
        # Get decision function parameters
        w = svm.coef_.flatten()
        b = svm.intercept_[0]
        
        # Store the decision function in the current node
        current_node.decision_function = f"w={np.round(w, 3)}, b={round(b, 3)}"
        
        # Project points onto decision axis
        projections = np.dot(remaining_X, w) + b
        
        # Sort projections
        sorted_indices = np.argsort(projections)
        sorted_projections = projections[sorted_indices]
        sorted_labels = remaining_y[sorted_indices]
        
        # Identify pure regions
        left_class = sorted_labels[0]
        right_class = sorted_labels[-1]
        
        leftmost_pure_end = 0
        while leftmost_pure_end < len(sorted_labels) and sorted_labels[leftmost_pure_end] == left_class:
            leftmost_pure_end += 1
        
        rightmost_pure_start = len(sorted_labels) - 1
        while rightmost_pure_start >= 0 and sorted_labels[rightmost_pure_start] == right_class:
            rightmost_pure_start -= 1
        
        # Remove pure regions
        left_pure_indices = sorted_indices[:leftmost_pure_end]
        right_pure_indices = sorted_indices[rightmost_pure_start + 1:]
        
        removed_left_X, removed_left_y = remaining_X[left_pure_indices], remaining_y[left_pure_indices]
        removed_right_X, removed_right_y = remaining_X[right_pure_indices], remaining_y[right_pure_indices]
        
        remaining_X = np.delete(remaining_X, np.concatenate((left_pure_indices, right_pure_indices)), axis=0)
        remaining_y = np.delete(remaining_y, np.concatenate((left_pure_indices, right_pure_indices)), axis=0)
        
        # Create tree nodes
        left_node = TreeNode(iteration=iteration_count, removed_class=left_class, removed_points=len(removed_left_y))
        right_node = TreeNode(iteration=iteration_count, removed_class=right_class, removed_points=len(removed_right_y))
        
        current_node.left = left_node
        current_node.right = right_node
        
        # Move to next iteration
        current_node = left_node if len(removed_left_y) > len(removed_right_y) else right_node
        
        if len(remaining_X) == 0:
            break
    
    return root

# Function to get tree dimensions
def get_tree_dimensions(root, depth=0):
    if root is None:
        return depth, 1
    
    left_depth, left_width = get_tree_dimensions(root.left, depth + 1)
    right_depth, right_width = get_tree_dimensions(root.right, depth + 1)
    
    return max(left_depth, right_depth), left_width + right_width

# Function to visualize the decision tree
def plot_tree(root, depth=0, x=0.5, y=0.9, dx=0.25, dy=0.1):
    if root is None:
        return
    
    # Draw the current node
    node_text = f"Iteration {root.iteration}"
    if root.removed_class is not None:
        node_text += f"\nRemoved: {root.removed_class}\nCount: {root.removed_points}"
    if root.decision_function:
        node_text += f"\n{root.decision_function}"
        
    plt.text(x, y, node_text,
             ha='center', va='center', bbox=dict(facecolor='lightblue', edgecolor='black'),
             wrap=True)
    
    # Calculate child positions
    if root.left:
        child_x = x - dx * (2 ** depth)  # Exponential spacing to avoid overlap
        child_y = y - dy
        plt.plot([x, child_x], [y, child_y], 'k-', lw=1)
        plot_tree(root.left, depth + 1, child_x, child_y, dx, dy)
    
    if root.right:
        child_x = x + dx * (2 ** depth)  # Exponential spacing to avoid overlap
        child_y = y - dy
        plt.plot([x, child_x], [y, child_y], 'k-', lw=1)
        plot_tree(root.right, depth + 1, child_x, child_y, dx, dy)

# Main execution
if __name__ == "__main__":
    X, y = load_dataset()
    tree_root = iterative_svm_refinement(X, y)
    
    # Calculate figure size based on tree dimensions
    max_depth, total_width = get_tree_dimensions(tree_root)
    fig_width = max(12, total_width * 3)  # Minimum width of 12, scale by tree width
    fig_height = max(8, max_depth * 1.5)  # Minimum height of 8, scale by tree depth
    
    plt.figure(figsize=(fig_width, fig_height))
    plt.title("Structured Iterative SVM Refinement Tree", pad=20)
    plot_tree(tree_root, dx=0.2)  # Adjust dx for better spacing
    plt.axis('off')
    plt.tight_layout()
    plt.show()
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

# Function to visualize the decision tree
def plot_tree(root, depth=0, x=0.5, y=0.9, dx=0.25, dy=0.1):
    if root is None:
        return
    
    # Draw the current node
    plt.text(x, y, f"Iteration {root.iteration}\nRemoved: {root.removed_class}\nCount: {root.removed_points}",
             ha='center', va='center', bbox=dict(facecolor='lightblue', edgecolor='black'))
    
    # Calculate child positions
    if root.left:
        child_x = x - dx * (depth + 1)  # Adjust x position based on depth to avoid overlap
        child_y = y - dy
        plt.plot([x, child_x], [y, child_y], 'k-', lw=1)
        plot_tree(root.left, depth + 1, child_x, child_y, dx, dy)
    
    if root.right:
        child_x = x + dx * (depth + 1)  # Adjust x position based on depth to avoid overlap
        child_y = y - dy
        plt.plot([x, child_x], [y, child_y], 'k-', lw=1)
        plot_tree(root.right, depth + 1, child_x, child_y, dx, dy)

# Main execution
if __name__ == "__main__":
    X, y = load_dataset()
    tree_root = iterative_svm_refinement(X, y)
    
    plt.figure(figsize=(12, 8))
    plt.title("Structured Iterative SVM Refinement Tree", pad=20)
    plot_tree(tree_root)
    plt.axis('off')
    plt.tight_layout()
    plt.show()
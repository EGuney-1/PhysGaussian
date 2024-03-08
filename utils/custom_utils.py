import json
import numpy as np
import torch

# # Specify the path to your JSON file
# file_path = "model/tree_4_30k/cameras.json"
# # Open the JSON file for reading
# with open(file_path, 'r') as file:
#     # Load the JSON data into a Python dictionary
#     data = json.load(file)

# positions = []

# for row in data:
#     positions.append(row["position"])

def fit_plane(points):
    # Compute centroid
    centroid = np.mean(points, axis=0)

    # Compute covariance matrix
    covariance_matrix = np.cov(points, rowvar=False)

    # Compute eigenvectors and eigenvalues
    eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)

    # Find the index of the smallest eigenvalue
    min_index = np.argmin(eigenvalues)

    # The normal vector to the plane is the eigenvector corresponding to the smallest eigenvalue
    normal = eigenvectors[:, min_index]

    # Ensure the normal vector is pointing outward from the centroid
    if np.dot(normal, centroid) > 0:
        normal *= -1

    return normal

def rotation_angles(normal):
    # Normalize the normal vector
    normal = np.array(normal) / np.linalg.norm(normal)
    
    # Compute rotation angle around x-axis
    theta_x = np.arctan2(normal[1], normal[2])
    
    # Compute rotation angle around y-axis
    theta_y = -np.arctan2(normal[0], np.sqrt(normal[1]**2 + normal[2]**2))
    
    # Compute rotation angle around z-axis (assuming z-axis is "up")
    theta_z = 0  # No rotation needed to make the normal upright
    
    return np.degrees(theta_x), np.degrees(theta_y), np.degrees(theta_z)

def min_max_values(vectors):
        # Convert the list of vectors to a PyTorch tensor
        tensor = torch.tensor(vectors)
        
        # Compute the minimum and maximum values along each dimension
        min_values, _ = torch.min(tensor, dim=0)
        max_values, _ = torch.max(tensor, dim=0)

        # Initialize an empty list to store the concatenated values
        boundary = []

        # Iterate over the range of the length of min_values (assuming min_values and max_values have the same length)
        for i in range(len(min_values)):
            boundary.append(min_values[i].item())
            boundary.append(max_values[i].item())

        return boundary

        


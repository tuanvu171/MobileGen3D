import json
import numpy as np

# Load the original JSON file
input_file_path = "/home/necphy/VuTruong/instruct-nerf2nerf/images-processed-colmap/images-park/transforms-original.json"
output_file_path = "/home/necphy/VuTruong/instruct-nerf2nerf/images-processed-colmap/images-park/transforms.json"

with open(input_file_path, "r") as json_file:
    data = json.load(json_file)

# Extract the list of poses and their transform matrices
poses = data["frames"]

# Convert the transform matrices to numpy arrays for easier computation
transform_matrices = [np.array(frame["transform_matrix"]) for frame in poses]

# Define a function to calculate the distance between two poses
def pose_distance(matrix1, matrix2):
    # Use the Euclidean distance between the translation parts of the matrices
    translation1 = matrix1[:3, 3]
    translation2 = matrix2[:3, 3]
    return np.linalg.norm(translation1 - translation2)

# Initialize the list to store selected poses
selected_poses = [poses[0]]  # Start with the first pose

# Set the desired number of final poses
desired_pose_count = 100

# Iterate through the poses and select the ones that are sufficiently far apart
for pose in poses[1:]:
    distances = [pose_distance(np.array(pose["transform_matrix"]), np.array(selected_pose["transform_matrix"])) for selected_pose in selected_poses]
    if all(dist > 0.5 for dist in distances):
        selected_poses.append(pose)

    # If we have reached the desired number of poses, stop
    if len(selected_poses) == desired_pose_count:
        break

# Create a new JSON data object with the selected poses
new_data = {
    "w": data["w"],
    "h": data["h"],
    "fl_x": data["fl_x"],
    "fl_y": data["fl_y"],
    "cx": data["cx"],
    "cy": data["cy"],
    "k1": data["k1"],
    "k2": data["k2"],
    "p1": data["p1"],
    "p2": data["p2"],
    "camera_model": data["camera_model"],
    "frames": selected_poses
}

# Write the new JSON data to the output file
with open(output_file_path, "w") as json_file:
    json.dump(new_data, json_file, indent=4)

print(f"Saved {len(selected_poses)} poses to {output_file_path}")
import kagglehub

folder_path = "LA/"

# Download the latest version (without specifying version number)
path = kagglehub.dataset_download("awsaf49/asvpoof-2019-dataset")

print("Path to dataset files:", path)

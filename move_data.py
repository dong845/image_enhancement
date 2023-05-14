import os
import shutil

def copy_file(source_path, destination_folder):
    shutil.copy(source_path, destination_folder)
    
output_dir = "merged_data"

train_high = output_dir+"/train/high"
train_low = output_dir+"/train/low"
test_high = output_dir+"/test/high"
test_low = output_dir+"/test/low"

def get_all_files(top_folder):
    for root, dirs, files in os.walk(top_folder):
        nums = int(len(files)*0.1)
        test_files = files[:nums]
        train_files = files[nums:]
        if "high_quality" in root:
            for file in train_files:
                if "png" in file:
                    file_path = os.path.join(root, file)
                    copy_file(file_path, train_high)
            for file in test_files:
                if "png" in file:
                    file_path = os.path.join(root, file)
                    copy_file(file_path, test_high)
        else:
            for file in train_files:
                if "png" in file:
                    file_path = os.path.join(root, file)
                    copy_file(file_path, train_low)
            for file in test_files:
                if "png" in file:
                    file_path = os.path.join(root, file)
                    copy_file(file_path, test_low)

get_all_files("/Users/lyudonghang/image_enhancement/train_datasets")
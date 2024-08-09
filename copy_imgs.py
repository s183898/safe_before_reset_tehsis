import os
import shutil
from collections import defaultdict


# Define source and destination directories
# source_dir = 'static/images/'

# step = 3 

# # Function to copy images
# def copy_dimmer(dest_dir, idx=0):

#     for filename in os.listdir('static/images'):
#         if filename.endswith(".png"):  # Check the file extension if needed
#             sample = filename.split('_')[-1]

#             if sample == "49.png":
#                 src_path = os.path.join(source_dir, filename)
#                 dest_path = os.path.join(dest_dir, filename)
#                 shutil.copy(src_path, dest_path)

            # if step*idx <= int(sample.strip(".png")) < ((idx+1)*step):    
            #     # Only copy the first two images for each combination
            #     src_path = os.path.join(source_dir, filename)
            #     dest_path = os.path.join(dest_dir, filename)
            #     shutil.copy(src_path, dest_path)

# copy_dimmer('static/intro_imgs')

# for i in range(10):
#     print("Here")
#     dest_dir = f'static/images_{i}'
#     os.makedirs(dest_dir, exist_ok=True)
#     copy_dimmer(dest_dir, i)


if __name__ == '__main__':
    
    if False:
        import os
        import shutil

        # Define the root folder
        root_folder = 'static/a_photo_of_a_nurse'

        # Get the list of method folders
        methods = [f for f in os.listdir(root_folder) if os.path.isdir(os.path.join(root_folder, f))]

        # Iterate over each method folder
        for method in methods:
            method_folder = os.path.join(root_folder, method)
            
            # Get the list of sample folders within each method folder
            samples = [f for f in os.listdir(method_folder) if os.path.isdir(os.path.join(method_folder, f))]
            
            # Iterate over each sample folder
            for sample in samples:
                sample_folder = os.path.join(method_folder, sample)
                
                # Define the original file path
                orig_file_path = os.path.join(sample_folder, 'orig.png')
                
                # Define the new file name and path
                new_file_name = f"{method}_{sample}.png"
                new_file_path = os.path.join(root_folder, new_file_name)
                
                # Move and rename the file
                shutil.move(orig_file_path, new_file_path)

        print("Files have been renamed and moved successfully.")


import os
from PIL import Image

src_folder = '/Users/juliagraham/MasterThesis/DatasetBackground'
dest_folder = '/Users/juliagraham/MasterThesis/DatasetRoboflow'

IGNORE_FILES = ['.DS_Store']

def main():
    for obj_folder in os.listdir(src_folder):
        if obj_folder in IGNORE_FILES:
            continue
        for file in os.listdir(os.path.join(os.path.join(src_folder, obj_folder))):
            if file in IGNORE_FILES:
                continue
            img = Image.open(os.path.join(os.path.join(src_folder, obj_folder), file))

            img.save(dest_folder + '/' + obj_folder + '_' + file)

        print("converted files: ", len(os.listdir(dest_folder)))

if __name__ == '__main__':
    main()
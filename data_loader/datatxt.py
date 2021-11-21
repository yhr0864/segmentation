import os
import glob
from tqdm import tqdm


class Datatxt:
    def __init__(self, root_path1: str, root_path2: str):
        self.save_dir = os.path.normpath(root_path1)
        self.save_dir2 = os.path.normpath(root_path2)

    def generateTxt(self):
        if not os.path.exists(self.save_dir):
            os.mkdir(self.save_dir)
            print(f'create folder: {self.save_dir}')

        # change the path for saving the txt
        with open(r'C:\Users\Myth\Desktop\data\train.txt', 'w') as f:
            path1 = self.save_dir + '\*png'
            path2 = self.save_dir2 + '\*png'

            pictures_list1 = glob.glob(path1)
            pictures_list2 = glob.glob(path2)

            with tqdm(total=len(pictures_list1)) as bar:
                idx = 0
                for pic in pictures_list1:

                    name = os.path.basename(os.path.normpath(pic))
                    label = os.path.basename(
                        os.path.normpath(pictures_list2[idx]))

                    f.write(os.path.join(self.save_dir, name) + ' ' +
                            os.path.join(self.save_dir2, label) + '\n')
                    idx = idx + 1
                    bar.update(1)

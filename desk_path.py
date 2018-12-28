import getpass
import os

# WorkSpace or Desktop
# and folder or file
# e.g. Desktop/Demo.py


def DeskPath(f_path):
    return "C:/Users/{}/Desktop/{}".format(getpass.getuser(), f_path)


def get_filename(path):
    file_triad = os.walk(path)
    all_file = []
    for dirpath, dirname, filename in file_triad:
        if filename != []:
            filename = [os.path.join(dirpath, f) for f in filename]
            all_file.extend(filename)
            print(dirpath)
    return all_file

def i_read(filename):
    pass


def i_save(fo, path):
    pass


print(find_file(DeskPath('SAT11/application/fuhs')))
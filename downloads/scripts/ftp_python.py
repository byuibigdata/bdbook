# %%
import ftplib, os, socket, sys, zipfile, traceback

HOST = r'ftp2.census.gov'
DRN = r'/geo/tiger/TIGER2020/BG/'
WRKSPCE_LOCAL = r'../'

f = ftplib.FTP(HOST)
f.login()
f.cwd(DRN)
filenames = f.nlst()
print(filenames)


# %%
fnames = ['tl_2020_16_bg.zip'] # we just need idaho
for ftt in fnames:
    with open(os.path.join(WRKSPCE_LOCAL, fnames[0]), 'wb') as local_file:
        f.retrbinary('RETR '+ ftt, local_file.write)
# %%

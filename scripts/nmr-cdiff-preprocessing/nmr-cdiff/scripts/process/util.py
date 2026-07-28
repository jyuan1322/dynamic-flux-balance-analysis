import glob
import os
import sys

def get_titles(fpath):
    tdata = []
    titles = []
    for tfile in glob.iglob(f"{fpath}/*/pdata/1/title"):
        tdata.append((tfile.split('/')[-4], tfile))
    for fid, tfile in sorted(tdata, key=lambda x: int(x[0])):
        with open(tfile, 'r') as rf:
            titles.append(f"# {fid} : {rf.readline().strip()}")
    return titles

def write_titles(titles):
    with open(f"{fpath}/titles.txt", 'w') as wf:
        for line in titles:
            wf.write(line + '\n')

def print_titles(titles, prefix=''):
    for line in titles:
        print(f"{prefix}{line}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        fpath = sys.argv[1]
    else:
        fpath = os.getcwd()
    fpath = fpath.rstrip('/')
    print("Reading titles...")
    titles = get_titles(fpath)
    print("Printing titles...")
    print_titles(titles, prefix="\t")
    print("Writing titles to titles.txt....")
    write_titles(titles)
    print("Done")

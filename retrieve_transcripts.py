import glob
import pandas as pd
# %%
transcript_files = glob.glob("/home/nakata/jsut_ver1.1_original/**/transcript_utf8.txt")
# %%
for transcript in transcript_files:
    with open(transcript, mode='r') as f:
        lines = f.readlines()
    for line in lines:
        filename, text = line.split(':')
        with open('raw_data/JSUT/JSUT/' + filename + '.lab', mode='w') as f:
            f.write(text.strip('\n'))

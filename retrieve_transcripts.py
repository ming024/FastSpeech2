import glob
import pandas as pd
import os
# %%
transcript_files = glob.glob("/media/ssd/corpus/jsut_ver1.1/*/transcript_utf8.txt")
# %%
if not os.path.exists("raw_data/JSUT/JSUT"):
    os.makedirs("raw_data/JSUT/JSUT")
for transcript in transcript_files:
    with open(transcript, mode='r') as f:
        lines = f.readlines()
    for line in lines:
        filename, text = line.split(':')
        with open('raw_data/JSUT/JSUT/' + filename + '.lab', mode='w') as f:
            f.write(text.strip('\n'))

import sys
import os
from pydub import AudioSegment

input = sys.argv[1]
fp_pfx = sys.argv[2]
time = int(sys.argv[3])  # in minutes

with open(input) as f:
    data = [x for x in f]

if fp_pfx == "none":
    fpaths = [x.split("|")[0] for x in data]
else:
    fpaths = [os.path.join(fp_pfx, x.split("|")[0] + ".wav") for x in data]

total_time = 0
new_data = []
for i, fp in enumerate(fpaths):
    audio = AudioSegment.from_wav(fp)
    total_time += len(audio) / 60000  # ms to min
    new_data.append(data[i])
    if total_time >= time:
        print(f"Total time is {total_time} minutes")
        break

fn, ext = os.path.splitext(input)

if (time / 60) < 1:
    timestamp = str(time) + "m"
else:
    timestamp = str(int(time / 60))

with open(fn + "_" + timestamp + ext, "w") as f:
    f.write("".join(new_data))

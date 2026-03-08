import sys
from pathlib import Path

KJ_TO_KCAL = 0.239005736

# Check command-line argument
if len(sys.argv) != 2:
    print("Usage: python FileConvert.py filename.xvg")
    sys.exit(1)

input_file = sys.argv[1]

# Make output filename automatically
input_path = Path(input_file)
output_file = "totalEint.dat"

with open(input_file, "r") as fin, open(output_file, "w") as fout:
    for line in fin:
        line = line.strip()

        # Skip comment/meta lines
        if not line or line.startswith("#") or line.startswith("@"):
            continue

        cols = line.split()
        if len(cols) < 3:
            continue

        time_ps = float(cols[0])
        coul_kj = float(cols[1])
        vdw_kj = float(cols[2])

        # Convert units
        time_ns = time_ps / 1000.0
        total_kcal = (coul_kj + vdw_kj) * KJ_TO_KCAL

        fout.write(f"{time_ns:.2f}\t{total_kcal:.8f}\n")
        

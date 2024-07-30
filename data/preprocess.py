from Bio import SeqIO
from tqdm import tqdm
import pandas as pd
import glob
import csv
import re


def generate_dot_bracket(sequence, base_pairs):
    dot_bracket = ["."] * len(sequence)
    for i, j in base_pairs:
        dot_bracket[i - 1] = "("
        dot_bracket[j - 1] = ")"

    return "".join(dot_bracket)


def process_file(file):
    with open(f"./data/{file}.dp") as f:
        lines = f.readlines()
    lines = [line.strip() for line in lines]
    lines = [
        line
        for line in lines
        if re.search(r"# file", line)
        or re.search(r"^[guac][guac]+[guac]$", line)
        or re.search(r"^[().][().]+[().]$", line)
    ]
    seq = []
    str = []
    Sequences = []
    Structures = []
    Files = []
    for line in tqdm(lines):
        if re.search(r"# file", line):
            Sequences.append("".join(seq))
            Structures.append("".join(str))
            Files.append(line)
            seq = []
            str = []
        if re.search(r"^[guac][guac]+[guac]$", line):
            seq.append(line)
        elif re.search(r"^[().][().]+[().]$", line):
            str.append(line)
    Sequences.append("".join(seq))
    Structures.append("".join(str))

    Sequences = Sequences[1:]
    Structures = Structures[1:]
    lengths = [len(sequence) for sequence in Sequences]
    df = pd.DataFrame(
        {
            "RNA_id": Files,
            "sequence": Sequences,
            "structure": Structures,
            "length": lengths,
            "source": file,
        }
    )
    for i in range(len(df)):
        df.at[i, "RNA_id"] = df.at[i, "RNA_id"].split(" ")[-1]
    df = df.drop(df[df["sequence"].str.len() == 0].index)
    df = df.drop(df[df["structure"].str.len() == 0].index)
    df = df.drop(df[df["sequence"].str.len() != df["structure"].str.len()].index)
    df = df.reset_index(drop=True)
    df.to_csv(f"./data/csv/{file}.csv", index=False)


def process_single(folder, files):
    mismatch = 0
    file_list = glob.glob(f"./data/{folder}/*.{files}")
    print("Total files: ", len(file_list))
    with open(f"./data/csv/{folder}.csv", "w") as f:
        writer = csv.writer(f)
        writer.writerow(["RNA_id", "sequence", "structure", "length", "source"])
        for file in tqdm(file_list):
            with open(file, "r") as f2:
                lines = f2.readlines()
                id = lines[0].strip().split(" ")[1]
                length = int(re.search(r"\d+", lines[1]).group())
                sequence = lines[3].strip()
                structure = lines[4].strip()
                if (
                    len(sequence) != len(structure)
                    or len(sequence) != length
                    or len(structure) != length
                ):
                    mismatch += 1
                else:
                    writer.writerow([id, sequence, structure, length, folder])
    print("Mismatch: ", mismatch)


def process_split(folder, file1, file2):
    mismatch = 0
    filelist1 = glob.glob(f"./data/{folder}/*.{file1}")
    print("Total files: ", len(filelist1))
    with open(f"./data/csv/{folder}.csv", "w") as f:
        writer = csv.writer(f)
        writer.writerow(["RNA_id", "sequence", "structure", "length", "source"])
        for file in tqdm(filelist1):
            with open(file, "r") as f1:
                records = SeqIO.parse(f1, "fasta")
                for record in records:
                    id = record.id
                    sequence = str(record.seq)
                    base_pairs = []
                    with open(
                        file.replace(file1, file2).replace("fasta", "bps"), "r"
                    ) as f2:
                        lines = f2.readlines()
                        for line in lines[2:]:
                            line = line.strip().split("\t\t")
                            base_pairs.append((int(line[0]), int(line[1])))
                    try:
                        structure = generate_dot_bracket(sequence, base_pairs)
                        writer.writerow(
                            [id, sequence, structure, len(sequence), folder]
                        )
                    except:
                        mismatch += 1
    print("Mismatch: ", mismatch)


process_file("RNAstrand")
process_split("TR0", "fasta", "bps")
process_split("TS0", "fasta", "bps")
process_split("VL0", "fasta", "bps")
process_split("TR1", "fasta", "bps")
process_split("TS1", "fasta", "bps")
process_split("VL1", "fasta", "bps")
process_single("bpRNA", "dbn")

df = pd.read_csv("./data/csv/bpRNA.csv")
df = df[["RFAM" in id for id in df["RNA_id"]]]
df.reset_index(drop=True, inplace=True)
df.drop_duplicates(subset=["sequence", "structure"], inplace=True)
df.reset_index(drop=True, inplace=True)
print("Filtered (RFAM): ", df.shape[0])
df.to_csv("./data/csv/RFAM.csv", index=False)

df = pd.read_csv("./data/csv/bpRNA.csv")
df = df[["CRW" in id for id in df["RNA_id"]]]
df.reset_index(drop=True, inplace=True)
df.drop_duplicates(subset=["sequence", "structure"], inplace=True)
df.reset_index(drop=True, inplace=True)
print("Filtered (CRW): ", df.shape[0])
df.to_csv("./data/csv/CRW.csv", index=False)

df = pd.read_csv("./data/csv/bpRNA.csv")
df = df[["PDB" in id for id in df["RNA_id"]]]
df.reset_index(drop=True, inplace=True)
df.drop_duplicates(subset=["sequence", "structure"], inplace=True)
df.reset_index(drop=True, inplace=True)
print("Filtered (PDB): ", df.shape[0])
df.to_csv("./data/csv/PDB.csv", index=False)

import pandas as pd
import glob

file_list = glob.glob("./data/csv/*.csv")


def append_files(file_list):
    df = pd.read_csv(f"./data/final/{file_list[0]}.csv")
    for file in file_list[1:]:
        df = pd.concat([df, pd.read_csv(f"./data/final/{file}.csv")], ignore_index=True)
    df.drop_duplicates(subset=["sequence", "structure"], inplace=True)
    df.reset_index(drop=True, inplace=True)
    df.drop_duplicates(subset=["sequence"], inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


for file in file_list:
    df = pd.read_csv(file)
    print("Original ({}): ".format(file), df.shape[0])
    df.drop_duplicates(subset=["sequence", "structure"], inplace=True)
    df.drop_duplicates(subset=["sequence"], inplace=True)
    df.reset_index(drop=True, inplace=True)
    print("Filtered ({}): ".format(file), df.shape[0])
    df.to_csv(file, index=False)

split_list = ["CRW", "RFAM", "RNAstrand", "PDB", "bpRNA"]

for file in split_list:
    df = pd.read_csv(f"./data/csv/{file}.csv")
    print("Original ({}): ".format(file), df.shape[0])
    train_df = df.sample(frac=0.75, random_state=42)
    test_df = df.drop(train_df.index)
    train_df.reset_index(drop=True, inplace=True)
    test_df.reset_index(drop=True, inplace=True)
    print("Train ({}): ".format(file), train_df.shape[0])
    print("Test ({}): ".format(file), test_df.shape[0])
    train_df.to_csv(f"./data/final/{file}-train.csv", index=False)
    test_df.to_csv(f"./data/final/{file}-test.csv", index=False)

remaining_list = ["TR0", "TR1", "VL0", "VL1", "TS0", "TS1"]

for file in remaining_list:
    df = pd.read_csv(f"./data/csv/{file}.csv")
    df.to_csv(f"./data/final/{file}.csv", index=False)

train_list = [
    "CRW-train",
    "RFAM-train",
    "RNAstrand-train",
    "PDB-train",
    "bpRNA-train",
    "TR0",
    "TR1",
    "VL0",
    "VL1",
]

test_list = [
    "CRW-test",
    "RFAM-test",
    "RNAstrand-test",
    "PDB-test",
    "bpRNA-test",
    "TS0",
    "TS1",
]

combined_df = append_files(train_list)
combined_df.to_csv("./data/final/train.csv", index=False)
print("Train: ", combined_df.shape[0])

combined_df = append_files(test_list)
combined_df.to_csv("./data/final/test.csv", index=False)
print("Test: ", combined_df.shape[0])

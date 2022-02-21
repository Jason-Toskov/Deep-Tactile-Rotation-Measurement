import pandas as pd
import glob

weights = {
    "breadboard": 85,
    "green_spray": 48,
    "magnet": 32,
    "pill": 28,
    "sanitizer": 123,
    "shampoo": 97,
    "swiss": 137,
    "toothbrush": 34,
    "toothpaste": 53,
    "usbc": 27,
}

INPUT_PATH = "./augmented_combined_dataset/"
OUTPUT_PATH = "./weight_norm_augmented_combined_dataset/"

count = 0
for dir in glob.glob(INPUT_PATH + "*.csv"):
    print(dir)
    w = None
    for k in weights.keys():
        if k in dir:
            w = weights[k]
            # print("True!")

    if w is not None:
        df = pd.read_csv(dir)
        # print(len(df))
        # count = 0
        for col in  df:
            if "fX" in col or "fY" in col or "fZ" in col:
                # count += 1
                # print(w)
                df[col] /= w
        # print(count)
        df.to_csv(OUTPUT_PATH + "weightdiv_" + dir.split("/")[-1], index=False)

    else:
        print("weird path name?")
        count += 1

print("Finished! Num failures = ", count)
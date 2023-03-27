import pandas as pd
import argparse

def convertFlairOutputToTSV(infile, outfile):
    df = pd.read_csv(infile, header=None, delim_whitespace=True)

    df.values

    df2 = df.drop([1], axis=1)

    # back to tsv
    df2.to_csv(outfile, sep='\t', index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--in-file", type=str, required=True,
                        help="Path to vertical format file output from Flair")
    parser.add_argument("--out-file", type=str, required=True,
                        help="Path to output file")
    args = parser.parse_args()

    convertFlairOutputToTSV(args.in_file, args.out_file )

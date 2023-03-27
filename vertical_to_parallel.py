import argparse

def convert_vertical_to_parallel(vertical_file, source_file, target_file):
    with open(vertical_file, "r") as f:
        lines = f.readlines()

    with open(source_file, "w") as f_src, open(target_file, "w") as f_tgt:
        curr_sent = []
        curr_labels = []

        for line in lines:
            if line.strip() == "":
                # End of sentence
                f_src.write(" ".join(curr_sent) + "\n")
                f_tgt.write(" ".join(curr_labels) + "\n")
                curr_sent = []
                curr_labels = []
            else:
                # Token and label in vertical format
                token, label = line.strip().split("\t")
                curr_sent.append(token)
                curr_labels.append(label)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--vertical-file", type=str, required=True,
                        help="Path to vertical format TSV file")
    parser.add_argument("--source-file", type=str, required=True,
                        help="Path to output source file")
    parser.add_argument("--target-file", type=str, required=True,
                        help="Path to output target file")
    args = parser.parse_args()

    convert_vertical_to_parallel(args.vertical_file, args.source_file, args.target_file)

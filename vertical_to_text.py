import argparse

def convert_vertical_to_text(vertical_file, output_file):
    with open(vertical_file, "r") as f:
        lines = f.readlines()

    with open(output_file, "w") as f_out:
        curr_sent = []

        for line in lines:
            print(line)
            if line.strip() == "":
                # End of sentence
                f_out.write(" ".join(curr_sent) + "\n")
                curr_sent = []
            else:
                # Token in vertical format
                token = line.strip()
                curr_sent.append(token)

        if curr_sent:
            # Write last sentence
            f_out.write(" ".join(curr_sent) + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--vertical-file", type=str, required=True,
                        help="Path to vertical format TSV file")
    parser.add_argument("--output-file", type=str, required=True,
                        help="Path to output text file")
    args = parser.parse_args()

    convert_vertical_to_text(args.vertical_file, args.output_file)

import argparse

def convert_wmt_to_vertical(source_file, prediction_file, output_file):
    with open(source_file, "r") as f_source, open(prediction_file, "r") as f_prediction, open(output_file, "w") as f_out:
        for source_sentence, prediction_sentence in zip(f_prediction, f_source):
            # Remove leading and trailing whitespace
            source_sentence = source_sentence.strip()
            prediction_sentence = prediction_sentence.strip()

            # Split source and prediction sentences into words
            source_words = source_sentence.split()
            prediction_words = prediction_sentence.split()

            # Write each word and its prediction to the output file
            for source_word, prediction_word in zip(source_words, prediction_words):
                f_out.write("{}\t{}\n".format(prediction_word, source_word))

            # Write a blank line to separate sentences
            f_out.write("\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-file", type=str, required=True,
                        help="Path to file containing source sentences")
    parser.add_argument("--prediction-file", type=str, required=True,
                        help="Path to file containing predicted translations")
    parser.add_argument("--output-file", type=str, required=True,
                        help="Path to output vertical format TSV file")
    args = parser.parse_args()

    convert_wmt_to_vertical(args.source_file, args.prediction_file, args.output_file)

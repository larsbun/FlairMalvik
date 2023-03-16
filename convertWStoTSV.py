import argparse, csv
import pandas as pd
# Construct the argument parser and parse the arguments
arg_desc = '''\
        Let's load an image from the command line!
        --------------------------------
            This program converts a file
            from whitespace sep to tsv with 
            Python argparse!
        '''
parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter,
                                 description=arg_desc)

parser.add_argument("-i", metavar="INPUT", help = "Path to your input file")
parser.add_argument("-o", metavar="OUTPUT", help = "Path to your output image")
args = vars(parser.parse_args())

if args["i"] and args["o"]:
    df = pd.read_csv(args["i"] ,
                     quoting=csv.QUOTE_NONE,
                     sep=' ',
                     header=None,
                     skip_blank_lines = False)

    df2=df.drop(df.columns[1], axis=1)
    df2.to_csv(args["o"], sep='\t',
               index=False,
               header=False,
               quoting=csv.QUOTE_NONE,)

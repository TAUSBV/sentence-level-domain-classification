import stanza
from argparse import ArgumentParser
from utils import read_file, write_lines_to_file


def main(filename, outfilename):

    # read data
    lines = read_file(filename)

    # create Stanza pipeline using the default model package
    nlp = stanza.Pipeline("en", processors="tokenize")

    sentences = list()

    # split lines into sentences
    for line in lines:
        doc = nlp(line)
        sents = [sent.text for sent in doc.sentences]
        for sent in sents:
            sentences.append(sent)

    # save results to file
    write_lines_to_file(outfilename, sentences)

    pass


if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("-in", "--infile")
    parser.add_argument("-out", "--outfile")
    args = parser.parse_args()

    main(args.infile, args.outfile)

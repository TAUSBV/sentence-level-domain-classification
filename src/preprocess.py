import glob
import stanza
from utils import read_file, write_lines_to_file


def main(filename, nlp, outfilename):

    # read data
    lines = read_file(filename)

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

    # create Stanza pipeline using the default model package
    nlp = stanza.Pipeline("en", processors="tokenize")

    # filepath is hard coded !
    for filename in glob.glob("bbc/*/*.txt"):
        print(filename)
        outfile = filename.replace(".txt", ".sents")
        main(filename, nlp, outfile)

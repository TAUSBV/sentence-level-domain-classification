import glob


def create_dataset():
    """
    Create a labeled data set from the sentence-level data.

    Note that this function will not work with a different folder structure,
    because it relies on extracting the labels based on the name of the directory
    each sentences file is contained in.
    """

    sentences = list()
    labels = list()
    # hard coding the filepath for the sake of simplicity
    for filename in glob.glob("bbc/*/*.sents"):
        sents = read_file(filename)
        for sent in sents:
            sentences.append(sent)
            # extracting labels based on directory name
            # if using a different data set or folder structure this lines needs to be changed appropriately
            labels.append(filename.split("/")[-2])

    return sentences, labels


def read_file(filename):
    """ Read a text file line by line and remove empty lines. """

    with open(filename, "r") as infile:
        contents = infile.readlines()
        contents = [line for line in [line.strip() for line in contents] if not len(line) == 0]

    return contents


def write_lines_to_file(outfilename, lines):
    """ Write lines in a list to a text file. """

    with open(outfilename, "w") as outfile:
        for line in lines:
            outfile.write(line)
            outfile.write("\n")

    pass

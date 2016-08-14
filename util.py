


def concatenatefiles(filenames,output_file_name):
    with open(output_file_name, 'w') as outfile:
        for fname in filenames:
            with open(fname) as infile:
                for line in infile:
                    outfile.write(line)



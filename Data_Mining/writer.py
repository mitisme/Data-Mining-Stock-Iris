#helper method to write data to txt files, used to see whats happening
def write_to_file(filename, information):
    #create the file, or see if it already exists
    with open(filename, 'w') as f:
        #write the information row by row, put blank line at the end
        for item in information:
            f.write(str(item) + '\n')
#end of write_to_file
            
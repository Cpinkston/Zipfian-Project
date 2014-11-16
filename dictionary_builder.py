import cPickle as pickle

FILE_DESTINATION ="/Users/CPinkston/Documents/Zipfian/FreedomOfSpeech/data" #sys.argv[2]

def create_dictionary(file):
    '''
    Input: Takes a file location where the text file created by CMU is located
    
    Output: dumps a pickle of that dictionary in FILE_DESTINATION
    
    turns the text into a dictionary and pickles it
    '''
    
    f = open(file, 'r')
    cmuDict = {}
    for line in f:
        listedline = line.strip('\n').replace('0','').replace('1','').replace('2','').split(' ')
        if len(listedline) > 1:
            cmuDict[listedline[0]] = listedline[2:]
        
    pickle.dump( cmuDict, open( FILE_DESTINATION + "/word_dict.p",'w') )
    
if __name__ == '__main__':
    create_dictionary('/Users/CPinkston/Downloads/cmudict.txt')
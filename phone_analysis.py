import os, sys
import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    TimitPath = r"C:\Users\camer\timit\TIMIT\TRAIN" #I want the raw string to signify \ as being just a character
    NonPhoneFile = 'NonPhonesList.txt'    #input
    CombinePhoneFile = 'CombinePhonesList.txt'   #input
    OutputPhoneProbFile = 'OutputPhoneProbs.txt' #output to be generated 
    OutputPhoneLengthFile = 'OutputPhoneLengthSec.txt' #output to be generated
    fs = 16000.0 # Hz

    # read in the TIMIT phone list
    filename = 'Phones.txt' #phones.txt has all 61 TIMIT phonemes
    with open(filename) as f:
        plist = f.readlines() #Return all lines in Phones.txt as a list where each line is an item in the list variable.
    phonelist = [] #create variable phonelist, empty list. 
    for i in plist:
        phonelist.append(i.strip()) #removes any leading & trailing whitespaces. Adds to phonelist.
    #phonelist now contains the 61 phonemes of TIMIT as entries to the list


    # create a dictionary (pair of two values associated with each other)
    #first value is the counter of PHN occurences, 2nd is for total sample length
    # One value for # observations, one value for accumulating the sample times   
    pdict = {}
    #create 61 dictionary keys and initialize the values in them to zero
    for i in phonelist:  
        pdict[i] = [0,0] #there are now 61 dictionary keys with two values per and initialized to zero

    # go through directories and find all .PHN files
    p=os.listdir(TimitPath)
    dirlist = [] 
    for i in p:
        if os.path.isdir(TimitPath + "\\" + i): #check if path is in directory
            dirlist.append(TimitPath +"\\" + i) #determine all paths, save them
        
    filelist = []
    for i in dirlist:
        p=os.listdir(i)
        for j in p:
            if os.path.isdir(i+'/'+j):
                subdir = i+'/'+j
                for file in os.listdir(subdir):
                    if file.endswith("PHN"):
                        filelist.append(i+'/'+j+'/'+file) #save all .PHN files to filelist
    #thus filelist contains all .PHN files as its entries

    # go through the PHN files and gather statistics
    for i in filelist: #for each .PHN file...
        with open(i) as f:
            plist = f.readlines()  #read info from the given .PHN file currently open
        for j in plist: #for all the lines in each file
            nline = j.strip() #removes any leading & trailing whitespaces (Characters in the middle of the string remain unaffected).
            fline = nline.split(" ") #split by the internal spaces. 
            #Each .PHN file has the following: starting sample  final sample    PHN
            pdict[fline[2]][0] += 1 # for key = the PHN, access value 0, add 1
            pdict[fline[2]][1] += (int(fline[1])-int(fline[0])) #make 1st and 2nd elements integers, subtract to get sample length
    



    #------------------------------------SECTION II: MAPPING -------------------------------------------------
   
    # combine these phones together
    # make two different phones into one, so add times AND occurrences
    # we will exclude the 2nd phone in the line, combining with 1st
    excludephones = []

    if os.path.exists(CombinePhoneFile):
        with open(CombinePhoneFile) as f:
            plist = f.readlines()
            for j in plist:
                nline = j.strip() #remove /n as well as whitepace
                fline = nline.split(" ")
                excludephones.append(fline[1]) #we are excluding the PHN fline[1]
                pdict[fline[0]][0] += pdict[fline[1]][0] #add occurences together
                pdict[fline[0]][1] += pdict[fline[1]][1]
                #for the PHN fline[0] and fline[1] add there second values together [1] which are the times
    
    
    # get the non phones classifiers left after mapping. There is only one h#
    nonphones = []
    with open(NonPhoneFile) as f:
        plist = f.readlines()
        for j in plist:
            nonphones.append(j.strip()) #thus, nonphones will only have one entry h#
    #since silence occurs randomly in real life we don't need its metrics from the data. 
    #I will just let the avg time of silence state be 0.5 secs



    #------------------------------------SECTION III: STATS -------------------------------------------------

    # compute the stats of each phone
    phonestats = {}
    total = 0
    for i in phonelist:
        if i not in excludephones: #don't include excluded as that data has been included in the other PHNs
            if i not in nonphones: #don't include h#
                total += pdict[i][0] #sum up all occurences
    for i in phonelist:
        if i not in excludephones:
            if i not in nonphones:
                # probability of each phone
                phonestats[i] = [float(pdict[i][0])/total, 0.0] #total occurences of PHN i/all occurences
                #initialize the second value to 0

                # average length of each phone in SECONDS.../fs to put it in secs
                phonestats[i][1] = (float(pdict[i][1])/pdict[i][0])/fs
                # phonestats[i][1] = (Total Length of PHN i / total occurences)/fs 
    

    # write the stats to file
    with open(OutputPhoneProbFile, 'w') as f:
        for i in phonelist:
            if i not in excludephones:
                if i not in nonphones:
                    f.write(i)
                    f.write(' ')
                    f.write(str(phonestats[i][0]))
                    f.write('\n')
    with open(OutputPhoneLengthFile, 'w') as f:
        for i in phonelist:
            if i not in excludephones:
                if i not in nonphones:
                    f.write(i)
                    f.write(' ')
                    f.write(str(phonestats[i][1]))
                    f.write('\n') 


    #------------------------------------SECTION IV: DISPLAY STATS -------------------------------------------------

    # create a couple bar graphs
    phonecats=[]
    phoneprobs=[]
    phonelens=[]
    for i in phonelist:
        if i not in excludephones:
            if i not in nonphones:
                phonecats.append(i)
                phoneprobs.append(phonestats[i][0])
                phonelens.append(phonestats[i][1])
  
    # creating the dataset  
    fig = plt.figure(figsize = (10, 8))
    
    # creating the bar plot
    plt.subplot(2,1,1)
    plt.bar(phonecats, phoneprobs, color ='blue',
            width = 0.4)
    
    plt.xlabel("Phone")
    plt.ylabel("Prob")
    plt.title("Phone Dictionary Relative Probabilities")
    #plt.show()
    
    # creating the bar plot
    plt.subplot(2,1,2)
    plt.bar(phonecats, phonelens, color ='red',
            width = 0.4)
    
    plt.xlabel("Phone")
    plt.ylabel("Len(s)")
    plt.title("Phone Dictionary Average Lengths(s)")
    plt.show()

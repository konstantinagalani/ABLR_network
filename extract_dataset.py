import os
import numpy as np

class Get_data():

    def read_file(self):
        """
        Get data from files from ECML 2016 dataset and separate them into y,x and metafeatures
        """
        dataset = input("Which dataset? ")
        if dataset == "svm" : directory = "/home/konstantina/Desktop/Project/data/svm"
        elif dataset == "adaboost" : directory = "/home/konstantina/Desktop/Project/data/adaboost"

        N = len(os.listdir(directory)) # number of files
        if dataset == "adaboost":
            shape = 2 # first two values of the files in adaboost are the x vector
        else:
            shape = 6 # first six values of the files in svm are the x vector

        P = 22
        y = []
        x_all = []
        metadata_all = []

        for filename in os.listdir(directory):
            x = []
            metadata = []
            print(filename)
            f = open(directory +"/" + filename)
            lines = f.read().splitlines()
            L = len(lines) # how many lines in each file

            for l in range(L):
                line = lines[l]
                data = line.split()
                y.append(data[0]) # 1st value: accuracy = y
                p_prev = 0

                for i in range(1,len(data)):

                    if data[i][2] != ':' and float(data[i][0]) < shape: # if the number of the parameter <10 and the data is x
                        p_new = int(data[i][0])
                        while(p_new - p_prev > 1):
                            x.append('0')# some files are missing parameters, assuming they are 0
                            p_prev +=1
                        x.append(data[i][2:]) # take out the index before the value
                        p_prev = p_new

                    elif data[i][2] != ':' and float(data[i][0]) >= shape: # if the number of the parameter <10 and the data is metafeatures
                        
                        while(len(x) != (l+1)*shape):
                            x.append('0') # values missing between 0-shape
                            p_prev +=1

                        p_new = int(data[i][0])
                        while(p_new - p_prev > 1):
                            metadata.append('0')# some files are missing parameters, assuming they are 0
                            p_prev +=1
                        metadata.append(data[i][2:]) # take out i:
                        p_prev = p_new

                    else: # if the number of the parameter >=10 and the data is metadata
                        p_new = int(data[i][:2])# the number of the parameter in the line
                        while(p_new - p_prev != 1):
                            metadata.append('0')# some files are missing parameters, assuming they are 0
                            p_prev +=1
                        metadata.append(data[i][3:])# take out index before the value
                        p_prev = p_new

                while (len(metadata)!= (l+1)*P):
                    metadata.append('0') # if the parameters missing are at the end of the line

            x = np.array(x).reshape((L,shape))
            metadata = np.array(metadata).reshape((L,P))
            x_all.append(x)
            metadata_all.append(metadata)

        x_all = np.array(x_all, dtype=float).reshape((N,L,shape))
        metadata_all = np.array(metadata_all, dtype=float).reshape((N,L,P))
        y = np.array(y, dtype=float).reshape((N,L))
        return y, x_all, metadata_all
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 31 14:10:44 2013

Seperate and manage data inputs and outputs

@author: Nelson
"""

import csv
import codecs

def ClearPrevResults(fileName):
    "clear previous result data"

    ofile  = open(fileName, 'w')
    write = csv.writer(ofile)
    ofile.close()    
   
    
    return
    
    
def ParameterImport(start,end,filename):
    
    #Import ParameterData from CSV file 
    inputfile  = open(filename, "rb")
    reader = csv.reader(codecs.iterdecode(inputfile, 'utf-8')) #row doesnt reset when called        
    
        
    "import all the data into array"
    rownum = 0    #counts number of rows
    data=[]    
    for row in reader:

        if rownum > start-1:
            if rownum < end+1:
                rowData=row           
                data.insert((rownum-1),rowData)
        rownum += 1        
    inputfile.close() 
    
    
    return [data,rownum]


def DataWriteThermo(ThermoResults_y):
    
    #Write Data (appending to end of  data file)
    ofile  = open('ThermoResults_y.csv', 'a')
    append = csv.writer(ofile)
    append.writerow(ThermoResults_y)
    ofile.close()
    
    return     


def PipeInputs():
   #Import PipeData from CSV file 
   inputfile  = open('PipeInputs.csv', "rb")
   reader = csv.reader(inputfile)
   rownum = 0
   for row in reader:
       if rownum == 1:
           data = row
       rownum += 1                
   inputfile.close()
   
   return data
   
   
def DataWriteSize(SizeResults):
   #Write Data (appending to end of  data file)
   ofile  = open('SizingResults_y.csv', 'a')
   append = csv.writer(ofile)
   append.writerow(SizeResults)
   ofile.close() 
   return


def DataWriteGens(filename,Gen):
    
    #Write Data (appending to end of  data file)
    ofile  = open(filename, 'a')
    append = csv.writer(ofile)
    append.writerow(Gen)
    ofile.close()
    
    return     


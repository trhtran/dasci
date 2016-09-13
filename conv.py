import os, os.path, sys
import pandas as pd
import re

counter=0

dirs = os.listdir('./xls')
for root, dirs, files in os.walk('./xls', topdown=True) :
    if counter>3: sys.exit()
    print root
    csvdir = re.sub('xls','csv',root)
    if not os.path.isdir(csvdir) :
        os.mkdir(csvdir)
    for file in files :
        counter = counter + 1
        if not file.endswith('.xls') : continue
        print 'XLS file: ', file
        baseFname,_ = os.path.splitext(file)
        print 'baseFname: ', baseFname
        #fullCsvFname = os.path.join(csvdir,csvFname)
        #print ' --> ',fullCsvFname

        fullXlsFname = os.path.join(root,file)
        xlsFile = pd.ExcelFile(fullXlsFname)
        sheets = xlsFile.sheet_names
        for sheet in sheets :
            newCsvFname = os.path.join(csvdir,'%s.%s.csv'%(baseFname,sheet))
            print 'creating ', newCsvFname

            df = xlsFile.parse(sheet,skiprows=5)
            df.to_csv(newCsvFname,sep=',',encoding='utf-8',index=False)



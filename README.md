# Telco Classification Project
This repo contains the files that were part of the telco classification project. The main files are the 'main' jupyter notebook, the predictions csv file, and the 'prepare' python file. This project also features a slide showing related visualizations, which can be found here: https://docs.google.com/presentation/d/1BgdEFfRdTrBau06xSG2ueQLgYIXm9gSLKRo1rHHFUXA/edit?usp=sharing

### How To Use The prepare.py File
It is very simple to prepare the data with this file. Simply have prepare.py in the working directory, import prepare into python, then call prepare.prep_telco(df). The 'df' in quotations is the variable that the data frame is stored in.

After using the prep_telco function, you will be able to split the data into train and test sets. To do this, call prepare.split_telco(df) and assign train and test to the output. 

Finally, you will want to normalize the data using min-max normalization. Do this by calling prepare.min_max_scale_telco(train, test), with 'train' and 'test' being the train and test data frames from the split function.
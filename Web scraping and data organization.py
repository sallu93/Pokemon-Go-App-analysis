# -*- coding: utf-8 -*-

import glob
from bs4 import BeautifulSoup
import datetime
import os
import gc
from pprint import pprint
import pandas as pd
from pandas import ExcelWriter



gc.enable()
data_dict=dict()
final_dict=dict()


os.chdir("C:/Users/Salman Mohammed/Desktop/Sem2 Spring 2017/1) Data Science INSY-5378-001/PROJECTS/Project 2/WD/data")

for folders in glob.iglob('*'):
    fol_date = datetime.datetime.strptime(folders, '%Y-%m-%d')
    min_counter = 0
    time_delta = datetime.timedelta(minutes=min_counter)
    #os.chdir("C:/Users/Salman Mohammed/Desktop/Sem2 Spring 2017/1) Data Science INSY-5378-001/PROJECTS/Project 2/pokemon_5378 (2)/data/" + folders)
    os.chdir("C:/Users/Salman Mohammed/Desktop/Sem2 Spring 2017/1) Data Science INSY-5378-001/PROJECTS/Project 2/WD/data/" + folders)
    print 'next folder'
    for filename in glob.iglob('*.html'):
        min_counter = 10
        time_delta = datetime.timedelta(minutes=min_counter)
        print filename
        with open(filename) as f:
            
            if 'android' in filename:
                print 'in android'
                
                soupA = BeautifulSoup(f , 'lxml' )
                # getting android star ratings
                star_ratings = soupA.find_all("span",{"class":"bar-number"})
                android_ratings_5 = star_ratings[0].get_text()
                android_ratings_4 = star_ratings[1].get_text()
                android_ratings_3 = star_ratings[2].get_text()
                android_ratings_2 = star_ratings[3].get_text()
                android_ratings_1 = star_ratings[4].get_text()
                
                #  getting total ratings
                tot_ratings = soupA.find_all("span",{"class":"reviews-num"})
                tot_ratings_value = tot_ratings[0].get_text()
                
                # getting average rating
                avg_rating = soupA.find_all("div",{"class":"score"})
                avg_rating_value = avg_rating[0].get_text()
                
                # getting file size
                file_size = soupA.find_all("div",{"itemprop":"fileSize"})
                file_size_value= file_size[0].get_text()

            elif 'ios' in filename:
                print 'in ios'
                soupI = BeautifulSoup(f , 'lxml' )
                # getting ratings from ios html files
                Customer_Ratings = soupI.find_all('span' , {'class':'rating-count'})
                ios_current_ratings= Customer_Ratings[0].get_text().split(" ")[0]
                ios_all_ratings= Customer_Ratings[1].get_text().split(" ")[0]
                
                # getting file size from ios html files
                File_Size = soupI.find_all('span' , {'class':'label'})
                ios_file_size= File_Size[3].parent.get_text().split(" ")[1]
                
                # putting data into dictionary
                final_dict[fol_date]={}
                    # ios data
                final_dict[fol_date]['ios_current_ratings']=ios_current_ratings
                final_dict[fol_date]['ios_all_ratings']=ios_all_ratings
                final_dict[fol_date]['ios_file_size']=ios_file_size
                    # Android data
                final_dict[fol_date]['android_avg_rating']=avg_rating_value
                final_dict[fol_date]['android_total_ratings']=tot_ratings_value
                final_dict[fol_date]['android_rating_1']=android_ratings_1
                final_dict[fol_date]['android_rating_2']=android_ratings_2
                final_dict[fol_date]['android_rating_3']=android_ratings_3
                final_dict[fol_date]['android_rating_4']=android_ratings_4   
                final_dict[fol_date]['android_rating_5']=android_ratings_5
                final_dict[fol_date]['android_file_size']=file_size_value
        
                pprint(final_dict)
                fol_date=fol_date+time_delta   
                print fol_date

print "making Pandas"
print final_dict.keys()
Pokemon_dataframe = pd.DataFrame(final_dict)
Pokemon_dataframe= Pokemon_dataframe.transpose()
print Pokemon_dataframe 

os.chdir("C:/Users/Salman Mohammed/Desktop/Sem2 Spring 2017/1) Data Science INSY-5378-001/PROJECTS/Project 2/WD")

with open('data.json' ,'w') as wjf:
    wjf.write(Pokemon_dataframe.to_json())
    
with open('data.csv' ,'w') as wcf:
    wcf.write(Pokemon_dataframe.to_csv())

writer = ExcelWriter('data.xlsx')
Pokemon_dataframe.to_excel(writer,'Sheet1')
writer.save()



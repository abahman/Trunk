# -*- coding: utf-8 -*-
"""
Created on Wed Dec 18 11:47:56 2019

@author: alsaibie
"""

from numpy import *
import pandas as pd


def create_course_schedule(semester = 'Spring 2020', class_start = '2020-01-19', class_end = '2020-04-30', final_exam_date='2020-05-18',
                           final_exam_time='0800-1000', class_days='Sun Tue Thu'):
    
    df = pd.DataFrame({"Date": pd.bdate_range(start=class_start, end=class_end, freq='C', weekmask=class_days)})

    df['Lecture'] = arange(1,df.shape[0]+1)
    df["Day"] = df.Date.dt.weekday_name
    # Reformat Date
    df['Date'] = df['Date'].apply(lambda x: x.strftime('%b %d'))
    # Crop to first three letters
    df["Day"] = df["Day"].str[:3]
    df['Topic'] = "           "
    df["Prior Reading Assignment"] = "       "
    df['Assignment'] = "         "
    df['Submission Due'] = "       "
    return df[['Lecture','Day','Date','Topic','Prior Reading Assignment','Assignment','Submission Due']]

writer = pd.ExcelWriter('Spring2020_Schedule_Template.xlsx', engine = 'xlsxwriter')
        
table24 = create_course_schedule(class_days='Mon Wed')
table24 = table24.set_index('Lecture')
table24.to_excel(writer, sheet_name='24')

table135 = create_course_schedule(class_days='Sun Tue Thu')
table135 = table135.set_index('Lecture')
table135.to_excel(writer, sheet_name='135')
writer.save()
writer.close()
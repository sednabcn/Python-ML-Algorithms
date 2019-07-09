#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 12 14:52:50 2019

@author: sedna
"""


import csv
with open('iris.data') as csv_file:
    csv_reader=csv.reader(csv_file, delimiter=',')
    line_count = 0
    for row in csv_reader:
        print( ','.join(row))

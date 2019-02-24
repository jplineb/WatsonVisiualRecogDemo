# -*- coding: utf-8 -*-
"""
Created on Sat Feb 16 18:07:31 2019

@author: jline
"""
from prettytable import PrettyTable
import pandas
import os
from watson_developer_cloud import VisualRecognitionV3

visual_recognition = VisualRecognitionV3(
        version='2018-03-19',
        iam_apikey = '6oqQup4aOd2a4TO2kbVzXgzH--4QXFgS-l8gcWc4hJIQ')

#acpt_imgformat = ('.png','.jpg')
#test_imgs = []

#==Slicks==#
Slicks_testing_set_folder = './test_set/Slicks/' # states the folder location that contains the testing images
Slicks_test_imgs = os.listdir(Slicks_testing_set_folder) #list test img names
Slicks_test_img_loc = [ Slicks_testing_set_folder + x for x in Slicks_test_imgs] #combines the directory and the image name into a list
amountSlicks = len(Slicks_test_img_loc)
#====================#

#==Treaded==#
Treaded_testing_set_folder = './test_set/Treaded/' # states the folder location that contains the testing images
Treaded_test_imgs = os.listdir(Treaded_testing_set_folder) #list test img names
Treaded_test_img_loc = [ Treaded_testing_set_folder + x for x in Treaded_test_imgs] #combines the directory and the image name into a list
amountTreaded = len(Treaded_test_img_loc)
#====================#

#==Classification==#
outputs= []
results = []
Amount_Classified_Correct = 0
Images_ran = 0
TP_count = 0
FN_count = 0
TN_count = 0
FP_count = 0

for z in Slicks_test_img_loc:
    with open(z , 'rb') as image_file: 
        classes = visual_recognition.classify(image_file, threshold= '0.6', owners=["me"]).get_result()
        outputs=(classes['images'][0])
        true_img_class = 'Slicks'
        predicted_img_class_watson = outputs['classifiers'][0]['classes'][0]['class']
        score = outputs['classifiers'][0]['classes'][0]['score']
        img_name = outputs['image']
        Images_ran = Images_ran + 1
        if true_img_class == predicted_img_class_watson:
            Amount_Classified_Correct = Amount_Classified_Correct + 1
            real_slick = 1
            predicted_slick = 1
            Truth_slick = 'TP'
            TP_count = TP_count + 1
        else:
            real_slick = 1
            predicted_slick = 0
            Truth_slick = 'FN'
            FN_count = FN_count + 1
        
        results.append([true_img_class,predicted_img_class_watson, score, Truth_slick , img_name])

for z in Treaded_test_img_loc:
    with open(z , 'rb') as image_file: 
        classes = visual_recognition.classify(image_file, threshold= '0.6', owners=["me"]).get_result()
        outputs=(classes['images'][0])
        true_img_class = 'Treaded'
        predicted_img_class_watson = outputs['classifiers'][0]['classes'][0]['class']
        score = outputs['classifiers'][0]['classes'][0]['score']
        img_name = outputs['image']
        Images_ran = Images_ran + 1
        if true_img_class == predicted_img_class_watson:
            Amount_Classified_Correct = Amount_Classified_Correct + 1
            real_treaded = 1
            predicted_treaded = 1
            Truth_treaded = 'TN'
            TN_count = TN_count + 1
        else:
            real_treaded = 1
            predicted_treaded = 0
            Truth_treaded = 'FP'
            FP_count = FP_count + 1
        results.append([true_img_class,predicted_img_class_watson, score, Truth_treaded, img_name])
#===================#

Accuracy =  (TP_count + TN_count) / (TP_count + TN_count + FN_count + FP_count) 
Precision = (TP_count)/(TP_count + FP_count)
Recall =(TP_count)/(TP_count + FN_count)
FPR = (FP_count)/(TN_count + FP_count)
Fscore = 2*(Precision*Recall)/(Precision + Recall)

alpha = PrettyTable()
alpha.field_names = ["True Image Class" , "Predicted Image Class", "Score", "Truth", "Image Name"]

for y in results:
    alpha.add_row(y)
    
print(alpha)

beta = PrettyTable()
beta.field_names = ["Accuracy", "Precision", "Recall", "Fscore"]
beta.add_row([Accuracy, Precision, Recall, Fscore])

print(beta)



#==to csv==#

columns = ["True Image Class" , "Predicted Image Class", "Score", "Truth", "Image Name"]
pd = pandas.DataFrame(results, columns = columns)
pd.to_csv("classifierresults.csv")

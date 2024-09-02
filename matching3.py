
# NAME: LOGESWARAN SIVAKUMAR
# template matching

# import all the required libraries packages
import cv2
import numpy as np
import argparse
import json
import os
from timeit import default_timer as timer
from skimage.io import imread_collection


# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-t", "--template", required=True, help="Path to template image")
ap.add_argument("-i", "--images", required=True, help="Path to images where template will be matched")
args = vars(ap.parse_args())

    
# key point & descriptor function    
def kp_des(coll_query, coll_train):
    
    print('******Running KP_DES******')
    # get image
    for img in coll_query:
        
        # find the keypoints and descriptors with SIFT
        kp_query, des_query = detector.detectAndCompute(img,None)
        kp_des_query.append((kp_query, des_query))
     
    # get template
    for img in coll_train:
        
        img_train = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # find the keypoints and descriptors with SIFT
        kp_train, des_train = detector.detectAndCompute(img_train,None)
        kp_des_train.append((kp_train, des_train))
    
    print('**********KP_DES************')
    return(kp_des_query, kp_des_train)

    

# define function for finding key matches
def find_matches(des_query, des_train, kp1, kp2):

    start1 = timer()
    key_matches = 0

    # FLANN parameters
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 10)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params,search_params)
    if(len(kp1)>=2 and len(kp2)>=2) :
        matches = flann.knnMatch(des_query, des_train, k=2)

    # Need to draw only good matches, so create a mask
    matchesMask = [[0,0] for i in range(len(matches))]

    # ratio test
    for i,(m,n) in enumerate(matches):
        if m.distance < 0.7*n.distance:
            matchesMask[i]=[1,0]
            key_matches = key_matches + 1

            # get the coordinates of the matching keypoints
            query_idx = m.queryIdx
            train_idx = m.trainIdx
            (x1,y1) = kp1[query_idx].pt
            (x2,y2) = kp2[train_idx].pt
            print("Coordinates of matching keypoints: ({}, {}) and ({}, {})".format(x1, y1, x2, y2))

    end1 = timer()
    print('find_match_time: ', (end1 - start1))
    return(key_matches, matches)


# match query image and template image function
def temp_query_match(coll_train, coll_query, kp_des_train, kp_des_query, query_name, train_name):
    
    print('******inside temp_query_match******')
    
    # run a loop through template images
    for i,template in enumerate(coll_train):
        
        print('------------------------------')
        print(train_name[i])
        print('------------------------------')
        
        if (train_name[i] == 'INSERT IMAGE NAMES THAT DOES NOT HAVE MANY FEATURES OR TOO SMALL'): #because this image data is causing problem, so skip it
            dicto['na'].append((train_name[i],[]))
            continue
        
            
        # get image and resize(to work with small template)
        trainImg = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
        
        # run a loop through images
        for j,imagePath in enumerate(coll_query):
            
            print('************************')
            print(query_name[j])
            print('************************')
            
            # get image and find the no. of matches using find_match()
            QueryImgBGR = imagePath
            key_matches, matches = find_matches(kp_des_query[j][1], kp_des_train[i][1], kp_des_query[j][0], kp_des_train[i][0])
            if matches == 0:
                dicto['na'].append((train_name[i],[]))
                continue
            
            # add image path to dictionary as key
            if query_name[j] not in dicto.keys():
                dicto.setdefault(query_name[j],[])
            
            # to check for major matches
            if key_matches > 70:
                
                # compute matches with distance less than 0.75
                goodMatch=[]
                for m,n in matches:
                    if(m.distance<0.55*n.distance):
                        goodMatch.append(m)
                
                # check if no. of matches is greater than your initialization and get template & query img keypts
                if(len(goodMatch)>MIN_MATCH_COUNT):
                    tp=[]
                    qp=[]
                    for m in goodMatch:
                        tp.append(kp_des_train[i][0][m.trainIdx].pt)
                        qp.append(kp_des_query[j][0][m.queryIdx].pt)
                    tp,qp=np.float32((tp,qp))
                    H,status=cv2.findHomography(tp,qp,cv2.RANSAC,3.0)
                    
                    # get the coordinates of corner pts and add it to dictionary
                    h,w=trainImg.shape
                    trainBorder=np.float32([[[0,0],[0,h-1],[w-1,h-1],[w-1,0]]])
                    if  H is not None:
                        queryBorder=cv2.perspectiveTransform(trainBorder,H)
                        cv2.polylines(QueryImgBGR,[np.int32(queryBorder)],True,(0,255,0),2)
                        print('queryborder: ', queryBorder)
                        print("Object found- %d/%d"%(len(goodMatch),MIN_MATCH_COUNT))
                        print(trainBorder)
                        dicto[query_name[j]].append(tuple((train_name[i],[int(queryBorder[0][0][0]),int(queryBorder[0][0][-1]), int(queryBorder[0][2][0]),int(queryBorder[0][2][-1])])))
                        print(dicto)
                        break
                else:
                    print ("Not Enough match found- %d/%d"%(len(goodMatch),MIN_MATCH_COUNT))
        
        # if no template has found match, add it to 'na' key in dictionary
        else:
            dicto['na'].append(tuple((train_name[i],[])))
            print(dicto)
    
    return(dicto)


# get names function
def load_images_from_folder(folder):
    name = []
    for filename in os.listdir(folder):
        name.append(filename)
    return name


# function main
def main():
    
    # get names of images
    train_name = load_images_from_folder(args["template"])
    query_name = load_images_from_folder(args["images"])
    # your path 
    col_dir_train = args["template"] + "/*.png;*.jpg;*.bmp"
    col_dir_query = args["images"] + "/*.png;*.jpg;*.bmp"
    # creating a collection with the available images
    coll_train = imread_collection(col_dir_train)
    coll_query = imread_collection(col_dir_query)
    kp_des_query, kp_des_train = kp_des(coll_query, coll_train)
    dicto = temp_query_match(coll_train, coll_query, kp_des_train, kp_des_query, query_name, train_name)
    # create a json file for dictionary
    with open('data.json', 'w') as file:
        json.dump(dicto, file, ensure_ascii=False, indent = 4)


# Sift object
detector=cv2.SIFT_create()

# initialize
MIN_MATCH_COUNT=60
dicto = {}
dicto.setdefault('na',[])
kp_des_query = []
kp_des_train = []
coll_train = []
coll_query = []
main()
            

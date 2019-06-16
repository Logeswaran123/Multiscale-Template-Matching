# Multiscale-Template-Matching
Multiscale Template Matching for multiple images and templates at the same time using OpenCV &amp; Python


## Approach

The main idea is to determine the key points and descriptors from images using SIFT and find matches based on the determined descriptors using Flann. The best matches are founded and checked if it's above a certain user defined threshold. If it's so, then the corner points os the template image is calculated and perspective transformation is applied to query image to get the border. The top-left corner and bottom-right corner of the border in query image are added to the dictionary in the corresponding key name.

If the number of matches are less than the user defined matches or if no matches can be found, then the template is compared with next query image and so on till it finds the first best match.

If no match can be found over entire query images data, then the template is added to the 'na' key value which is no template association.

## Flann Based Matcher

Flann is a faster and efficient way to find matches by clustering. Feature descriptors like SIFT, SURF use euclidean distance and Binary descriptor like ORB are matched using hamming distance. But, to reduce the number of false matches, Flann based matching is done by the distance ratio between the two nearest matches of a considered keypoint  and it is a good match when this value is below a thresold. Flann builds an efficient data structure (KD-Tree) that will be used to search for an approximate neighbour.

I tested the data with SIFT, SURF, ORB and Flann, and Flann seems to be the more accurate and efficient way to match the data, especially scaled image data.

## 1st approach

In this approach, I tried using canny edge detection and then matching the template using cv2.matchTemplate(). I varied the size of the templates from 10% to 200% till it could find the best match. But this approach could not yield good results as resolution of smaller templates are low.

## 2nd approach

In this approach, I tried feature matching with various algorithms like SIFT, SURF, ORB, BruteForce algorithms to find the features and match the image and template. But using BruteForce seemed  time consuming and since the data given to match is large, BruteForce wasn't the efficient way.

## 3rd approach

In this approach, I tried to increase the speed and accuracy by using SIFT for feature detection and Flann based matcher for matching the templates and images with the best number of matches. This gives 10 times better performance interms of speed. This is the approach I finally used for this task.

Note: Json file format doesn't support tuples, so tuples are converted as lists in the output json file.


Happy learning,
Logeswaran




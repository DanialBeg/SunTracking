
# coding: utf-8

# In[ ]:

import cv2
import numpy as np

cap = cv2.VideoCapture(0)

while(1):

   _, frame = cap.read()
   (h, s, v) = cv2.split(frame)
   s = s + 10;
   s = np.clip(s,0,255)
   frame = cv2.merge([h,s,v])
   shadowc = cv2.cvtColor(frame, cv2.COLOR_RGB2RGBA)
   edges = cv2.Canny(shadowc,100,200)
   cv2.imshow("Frame", frame)
   cv2.imshow("Edges", edges)
   k = cv2.waitKey(5) & 0xFF
   if k == 27:
       break

cv2.destroyAllWindows()


# In[2]:

import cv2
import numpy as np
from matplotlib import pyplot as plt
 
cap = cv2.VideoCapture(1)


while(1):
 
    _, frame = cap.read()
    (h, s, v) = cv2.split(frame)
    s = s + 4;
    s = np.clip(s,0,255)
    frame = cv2.merge([h,s,v])
    shadowc = cv2.cvtColor(frame, cv2.COLOR_RGB2RGBA)
    edges = cv2.Canny(shadowc,100,200)
    cv2.imshow("Frame", frame)
    cv2.imshow("Edges", edges)
    
    plt.subplot(122),plt.imshow(edges,cmap = 'gray', alpha = 1.0)
    plt.title('Edge Image'), plt.xticks([]), plt.yticks([]),     
    plt.imshow(frame,zorder=1, alpha=0.5)
   
    width = 16
    height = 9
    plt.figure(figsize=(width, height))
    get_ipython().magic(u'matplotlib')
    
    plt.show()
   
    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break
 
cv2.destroyAllWindows()


# In[ ]:

import cv2
import numpy as np
from matplotlib import pyplot as plt
 
cap = cv2.VideoCapture(0)
width = 16
height = 9
plt.figure(figsize=(width, height))
while(1):
 
    _, frame = cap.read()
    (h, s, v) = cv2.split(frame)
    s = s + 4;
    s = np.clip(s,0,255)
    frame = cv2.merge([h,s,v])
    shadowc = cv2.cvtColor(frame, cv2.COLOR_RGB2RGBA)
    edges = cv2.Canny(shadowc,100,200)
    cv2.imshow("Frame", frame)
    cv2.imshow("Edges", edges)
    get_ipython().magic(u'matplotlib')
    
    plt.subplot(122),plt.imshow(edges,cmap = 'gray', alpha = 1.0)
    plt.title('Edge Image'), plt.xticks([]), plt.yticks([]),     
    plt.imshow(frame,zorder=1, alpha=0.5)
   
 
    plt.show()
   
    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break
 
cv2.destroyAllWindows()


# In[ ]:

import cv2
import datetime
import numpy as np
from matplotlib import pyplot as plt
import time
i = 0
cap = cv2.VideoCapture(0)
now = ""
width = 16
height = 9
minutes = int(time.strftime("%m"))

nows = time.strftime("%s")
plt.figure(figsize=(width, height))
while True:
    flag, frame = cap.read()

    if flag == 0:
        break
    (h, s, v) = cv2.split(frame)
    s = s + 3;
    s = np.clip(s,0,255)
    frame = cv2.merge([h,s,v])
    shadowc = cv2.cvtColor(frame, cv2.COLOR_RGB2RGBA)
    edges = cv2.Canny(shadowc,100,200)
    cv2.imshow("Frame", frame)
    cv2.imshow("Edges", edges)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    diff = 7
    w1 = np.array([0,0,255-diff])
    w2 = np.array([255,diff,255])
    mask = cv2.inRange(hsv, w1, w2)
    # Cloudy day detector
    
    if np.any(frame[100, 100] < [100, 100, 100]):
        nowm = datetime.datetime.now().minute
        if (nowm != minutes):
            i = 0
        
        i = i + 1
        if (True):
            print("It is cloudy today. Expect some variance in data results. The time is currently %s. Last clouds reported at %s." % ((str(datetime.datetime.now().time())), str(now)))  
            now = time.strftime("%c")
            nows = time.strftime("%s")
            get_ipython().magic(u'matplotlib inline')
            for x in range(0, i+1):
                plt.scatter(datetime.datetime.now().minute,x)
            minutes = datetime.datetime.now().minute
            print minutes
            plt.axis([0, 60, 0, i*2])
            plt.xlabel('Minutes')
            plt.ylabel('Occurences')
            plt.show()
            res = cv2.bitwise_and(frame,frame, mask= mask)
    get_ipython().magic(u'matplotlib')
    
    plt.subplot(122),plt.imshow(edges,cmap = 'gray', alpha = 0.5)
    plt.title('Edge Image'), plt.xticks([]), plt.yticks([]),     
    plt.imshow(frame,zorder=1, alpha=1.0)
    plt.show()
    # Quit key is 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
cap.release()
cv2.destroyAllWindows()


# In[ ]:




# In[ ]:




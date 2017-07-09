
import urllib2
import urllib
import random
import logging
from PIL import Image, ImageFilter

threshold = 30

count = 1

def setcolor(ori,edge,color, c,x, y, n):

    flag = False
    if(x < 256 and y  < 256):
        if( x > 0 and y  >0):
            if edge[x,y] != (0,0,0):

                for i in range(-1,2):
                    for j in range(-1,2):

                        #print x+i
                        #print y+j
                        if(x+i < 256 and y +j < 256):
                            if( x+i > 0 and y +j >0):
                                if edge[x+i,y+j] == (255,255,255):
                                    color[x+i,y+j] = c
                                else:
                                    flag = False
                                    return flag
                flag = True
    #colorimg.save(str(n) + "_color.jpg")
    return flag


def create():
    global count

    for i in range(1,20000):
        try:
            imghttp = urllib2.urlopen("http://danbooru.donmai.us/posts/" + str(i)).read()

            line = imghttp.split("\n")

            for l in line:
                if l.find('/cached/data/') > -1 and l.find('<img width') > -1:
                    imglink = "http://danbooru.donmai.us" + l[l.find('src=')+5: -4]
                    resource = urllib.urlopen(imglink)
                    output = open("tmp.jpg","wb")
                    output.write(resource.read())
                    output.close()
                    output = open("color.jpg","wb")
                    output.write(resource.read())
                    output.close()

                    img1 = Image.open("tmp.jpg")
                    if img1.size[0] > img1.size[1]:
                        cut = img1.size[1]
                    else:
                        cut = img1.size[0]
                    img2 = img1.crop((0,0,cut,cut))
                    img3 = img2.resize((256,256),Image.ANTIALIAS)
                    img3.save(str(count)+"_ori.jpg")
                    img4 = img3.filter(ImageFilter.FIND_EDGES)
                    img4_fixed = img4.load()
                    for j in range(256):
                        for k in range(256):
                            r,g,b = img4_fixed[j,k]
                            img4_fixed[j,k] = (255 - r ,255 -g , 255 -b )
                            if r < threshold or g < threshold or b < threshold:
                                img4_fixed[j,k] = (255,255,255)
                            else:
                                img4_fixed[j,k] = (128,0,0)

                    img4.save(str(count)+"_edge.jpg")

                    ran = random.randint(10,20)
                    ran2 = random.randint(10,20)

                    colorimg = img3.copy()
                    color_pixel = colorimg.load()

            #        print (k)
                    for j in range(256):
                        for k in range(256):
                            color_pixel[j,k] = (0,0,0)


                    for l in range(ran):
                        x = random.randint(1,255)
                        y = random.randint(1,255)
                        c = img3.load()[x,y]

                        tmpx = x
                        tmpy = y

                        for m in range(ran2):
                            s = random.randint(0,4)


                            f = setcolor(img3.load(),img4.load(),color_pixel,c, tmpx, tmpy, i)

                            if f == False:
                                tmpx = x
                                tmpy = y
                                continue
                            if s == 0:
                                tmpx += 3
                            elif s == 1:
                                tmpx -= 3
                            elif s == 2:
                                tmpy += 3
                            elif s == 3:
                                tmpy -= 3


                    cp = colorimg.copy()
                    cp.save(str(count)+"_color.jpg")

                    count += 1

                    print imglink
                    print(i)
        except Exception:
            print("error")

create()

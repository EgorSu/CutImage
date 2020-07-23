import cv2
import numpy
import math
import random
#Только для тестов:
import os
from datetime import datetime


class Cl:
    image = 0 
    rows = 0 
    cols = 0
    dtime = 0
    def __init__(self,image_path):
        image = cv2.imread(image_path)
        self.image_path = image_path
        self.image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        self.rows = image.shape[0]
        self.cols = image.shape[1]
        #self.image = self.find_edges()
        #image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    def changeImage(self, image):
        self.image = image
        self.rows = image.shape[0]
        self.cols = image.shape[1]    
    def getimage(self, booll):    
        self.changeImage(self.find_angle())
        return self.FindAllMinimum(self.findHist(booll), booll) 
    def find_angle(self):
        time = datetime.now()
        gray = self.image[((self.rows//3)):(self.rows//3)+((self.rows//3)),((self.cols//3)):(self.cols//3)+((self.cols//3))]
        gray = cv2.medianBlur(gray,5)
        gray = cv2.GaussianBlur(gray,(5,5),9)
        y_average = self.croped_lines(True, gray)
        cols = gray.shape[1]
        rows = gray.shape[0]
        for i in range(len(y_average)): 
            val = [0]
            dVal = 0
            dValArr = []
            for j in range(1, cols):
                val.append(int(gray[y_average[i][0],j]))
                dVal = (val[j] - val[j - 1])
                dValArr.append(dVal)
        edges = cv2.Canny(gray,max(dValArr)//6,max(dValArr)//8)
        angle_list = []
        angle_dict = {}
        lines = cv2.HoughLines(edges,1,numpy.pi/180,max(dValArr),)
        step = 10 if 10<=len(lines) else len(lines)
        edges_lines = []
        for i in range(step):
            for rho,theta in lines[i]:
                a = numpy.cos(theta)
                b = numpy.sin(theta)
                x0 = a*rho
                y0 = b*rho
                x1 = int(x0 + 1000*(-b))
                y1 = int(y0 + 1000*(a))
                x2 = int(x0 - 1000*(-b))
                y2 = int(y0 - 1000*(a))
                edges_lines.append((x1,y1,x2,y2))
                if (x2 - x1) != 0:
                    angle_rad = math.atan2((y2 - y1),(x2 - x1))
                    angle = math.degrees(angle_rad)
                    if abs(angle) > 45:
                        angle = 90 + angle
                else:
                    angle = 0
                angle = float(f"{angle:.{3}f}")
                angle_list.append(angle)
        edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
        
        for num in angle_list:
            if num in angle_dict:
                angle_dict[num] += 1
            else:
                angle_dict[num] = 1
        angle_list = list(angle_dict.items())
        angle_list.sort(key=lambda i: i[1], reverse=True)
        avr_angle = float(angle_list[0][0])
        (h, w) = self.image.shape
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, avr_angle, 1.0)
        rotated = cv2.warpAffine(self.image, M, (w, h))
        step_t = 40
        x_table = self.cols//step_t
        y_table = self.rows//step_t
        if abs(avr_angle) > 5 :
            return rotated
        return rotated
    def croped_lines(self, booll,img):
        y_average = [] 
        cols = img.shape[1] 
        rows = img.shape[0]
        if not booll:
            for y in range(0, cols, (cols // 8)):
                sum_y = 0
                for x in range(rows):
                    sum_y += img[x,y]
                cort = (y , (sum_y / cols))
                y_average.append(cort)
        else:
            for x in range(0, rows, (rows // 8)):
                sum_x = 0
                for y in range(cols):
                    sum_x += img[x,y]
                cort = (x , (sum_x / cols))
                y_average.append(cort)
        y_average.sort(key=lambda i: i[1], reverse=True)
        return y_average[0:5]          
    def FindIndexMax(self, array):
        maxVal = 0
        index = 0
        for i in range(len(array)):
            if maxVal < array[i]:
                maxVal = array[i]
                index = i
        return index   
    def plotHist(self, array, name):
        hist = numpy.zeros((500, len(array))) #
        max_value = max(array)
        for i in range(len(array)):
            cv2.line(hist, (i, 500), (i, int((max_value - array[i]) * 500/max_value)), (255))
        return hist
    def plotHistZero(self, array, name):
        hist = numpy.zeros((1000, len(array)))#size len * 500
        max_value = max(array) if max(array) > abs(min(array)) else abs(min(array))
        for i in range(len(array)):
            cv2.line(hist, (i, 500), (i, int(500 - array[i] * 500/max_value)), (255))
        return hist    
    def plotHistLines(self, array, linesGray, name):
        hist = numpy.zeros((500, len(array)), numpy.uint8)#size len * 500
        max_value = max(array)    
        for i in range(len(array)):
            color = (255 - 30 * (ch % 6 + 1))
            if ch < len(linesGray) and linesGray[ch] < i:
                ch += 1            
            cv2.line(hist, (i, 500), (i, int((max_value - array[i]) * 500/max_value)), (color), 1)        
        return hist	
    def cutIMG(self, cutarray, image, border=0): # cutarray = [(x_c, y_c), (x_h, y_h)]
        result = cutarray
        for i in range(0, len(cutarray)):
            x_c, y_c = cutarray[i][0]
            x_h, y_h = cutarray[i][1]
            cropped = image[y_c : y_c + y_h, x_c : x_c + x_h]
            if border!=0:
                cv2.rectangle(cropped, (x_c, y_c), (x_h, y_h), 255, border)
                cv2.rectangle(image, (x_c, y_c), (x_h, y_h), 255, border)
            result[i].append(cropped)
            original_with = image
        return result, original_with_cut_lines  # = [(x_c, y_c), (x_h, y_h), [cut_image]]
    def findHist (self, booll):
        start_time = datetime.now()
        cols = self.cols
        rows = self.rows
        array = []    
        edge = self.two_lines(booll)
        edge.sort(key=lambda i: i[0])
        if booll:
            array = [0 for i in range(cols)]
            for i in range(edge[0][0], edge[1][0], (edge[1][0] - edge[0][0]) // 100):
                for j in range(cols):
                    array[j] = int(self.image[i,j]) 
        else:            
            arrayAV = [0 for i in range(rows)]
            for i in range(edge[0][0], edge[1][0], (edge[1][0] - edge[0][0]) // 100):
                for j in range(rows):
                    array[j] = int(self.image[j,i])
                
        return array
    def FindAllMinimum(self, array, booll): # find bar chart minimum

        # find start and end of bar chart's values
        start = 0
        end = 0
        startFind = False
        endFind = False
        th = max(array)
        rightZero = 0
        leftZero =  0
        findRightZero = False
        findLeftZero = False
        for i in range(len(array)//2):
            if not findLeftZero and array[len(array)//2 - i]  < 100:
                findLeftZero == True
                leftZero = len(array)//2 - i
            if not findRightZero and array[len(array)//2 + i] < 100:
                findRightZero == True
                rightZero = len(array)//2 + i    
        for i in range(len(array)//2):
            if array[i] > th/4 and not startFind:
                start = i
                startFind = True
            if array[len(array) - i - 1] > th/4 and not endFind:
                end = len(array) - i -1
                endFind = True
            if endFind and startFind:
                break
        start, startLeft  = self.FindPointSpace(start, array)
        endRight, end = self.FindPointSpace(end, array)     
        
        thresh = []
        pointArray = []
        average = 0
        thRange = 10
        # give gradient value of bar chart in range thRange
        for i in range(start if start > 20 else 20, end if len(array) - 20 > end else len(array) - 20):
            sumD = 0
            for j in range(thRange):
                sumD += array[i + j] - array[i + j - 1]   
            thresh.append(sumD) 
        #mean value filter with coeff           
        for i in range(thRange, len(thresh) - thRange):
            thresh[i] = (thresh[i - 1]//2 + thresh[i] + thresh[i + 1]//2)//3 #   
            if(thresh[i] > 0):
                average += thresh[i]
        average /= len(thresh)    
        avMult = 0.7
        pointsMax = [] 
        maxArr = []
        maxArr = self.findPike(thresh, int(average * avMult))
        indexMax = self.FindIndexMaxValAdditional(maxArr)
        pointsMax = self.FindDl(maxArr, indexMax)
        #        
        for i in range(len(pointsMax)):
            pointArray.append(pointsMax[i] + start + thRange - 20)  
            pointArray.append(pointsMax[i] + start + thRange) 
        pointArray.sort()            
        return self.plotHistLines(array, pointArray, 'result')
    def findPike(self, thresh, average):
        maxArr = []
        minArr = []
        isFindMax = False
        isFindMin = False
        i = 0
        while i < len(thresh):
            localMaxIndex = 0
            localMinIndex = 0
            localMaxVal = 0
            localMinVal = 0
            widht = 0
            endIndex = 0
            startIndex = 0
            while i < len(thresh) and thresh[i] > average:
                endIndex = i
                if localMaxVal < thresh[i]:
                    localMaxVal = thresh[i]
                    localMaxIndex = i
                isFindMax = True
                i += 1 
            if isFindMax:
                maxArr.append((localMaxIndex, thresh[localMaxIndex], endIndex)) 
                isFindMax = False
            #can use minArr for find widht or additional point array      
            '''    
            widht = 0    
            while i < len(thresh) and thresh[i] < -average:
                if(not isFindMin):
                    startIndex = i
                isFindMin = True
                if localMinVal > thresh[i]:
                    localMinVal = thresh[i]
                    localMinIndex = i
                    isFindMin = True
                i += 1
            if isFindMin:
                minArr.append((localMinIndex, thresh[localMinIndex], startIndex))
                isFindMin = False 
            '''    
            i += 1
        i = 0
        '''
        resultArray = []
        resultArray.append(( maxArr[0][0],(maxArr[0][2] - maxArr[0][0]) * 2 * maxArr[0][1]))
        lastPosition = 0    
        while i < (len(maxArr)):
            j = (len(minArr)) - 1
            xPosMaxVal = maxArr[i][0]
            while j > lastPosition :
                if minArr[j][0] < xPosMaxVal:
                    resultArray.append(((maxArr[i][2] + minArr[j][2])//2, (maxArr[i][2] - minArr[j][2]) * maxArr[i][1]))
                    lastPosition = j
                    break
                j -= 1
            i += 1
        resultArray.append( (minArr[len(minArr) - 1][0],(minArr[len(minArr) - 1][2] - minArr[len(minArr) - 1][0]) * 2 * minArr[0][1]))  
        '''
        return maxArr  
    def FindPointSpace(self, index, array):
        it = 1
        leftIndex = index
        rightIndex = index
        dArrRight = []
        dArrLeft = []
        while index + it == len(array) or index - it < 0: 
            dArrRight = array[index + it] - array[index + it - 1]
            dArrLeft = array[index - it] - array[index - it + 1]
            if dArrRight < 0 or rightIndex != index:
                rightIndex += it
            if dArrLeft < 0 or leftIndex != index:
                leftIndex -= it                    
            if (leftIndex != index and rightIndex != index) or it > 300:
                break     
            it += 1  
        return rightIndex, leftIndex    
        #-----------------
    def FindIndexMinVal(self,array):
        ch = 0
        minVal = 2000000000
        for i in range(len(array)):
            if array[i] < minVal:
                minVal = array[i]
                ch = i
        return ch
    def FindIndexMinValAdditional(self,array):
        ch = 0
        minVal = 2000000000
        for i in range(len(array)):
            if array[i][1] < minVal:
                minVal = array[i][1]
                ch = i
        return ch
    def FindIndexMaxVal(self,array):
        ch = 0
        maxVal = 0
        for i in range(len(array)):
            if array[i] > maxVal:
                maxVal = array[i]
                ch = i
        return ch    
    def FindIndexMaxValAdditional(self,array):
        ch = 0
        maxVal = 0
        for i in range(len(array)):
            if array[i][1] > maxVal:
                maxVal = array[i][1]
                ch = i
        return ch 
    def two_lines(self, booll):
        y_average = [] 
        cols = self.cols 
        rows = self.rows 
        if not booll:
            for y in range(0, cols, (cols // 8)):
                sum_y = 0
                for x in range(rows):
                    sum_y += self.image[x,y]
                cort = (y , (sum_y / cols))
                y_average.append(cort)
        else:
            for x in range(0, rows, (rows // 8)):
                sum_x = 0
                for y in range(cols):
                    sum_x += self.image[x,y]
                cort = (x , (sum_x / cols))
                y_average.append(cort)
        y_average.sort(key=lambda i: i[1], reverse=True)
        return y_average[0:5]
    def FindDl(self, array, startIndex):
        points = []
        dlArray = self.allDl(array, startIndex, len(array), 1) + self.allDl(array, startIndex, 0, -1)
        self.delRepeatElem(dlArray)
        wasFound = False
        i = 0
        for i in range(len(dlArray)): 
            if self.findNextMinimum(dlArray[i], array, array[startIndex], startIndex, points, 0) and self.findNextMinimum(-dlArray[i], array, array[startIndex], startIndex, points, 0):
                nextPoint = 0
                points.append(array[startIndex][0])
                points.sort()
                isNext = False #is it next or previous point
                for j in range(len(points)):
                    if points[j] == array[startIndex][0]:
                        if j >= len(points)//2:
                            nextPoint = points[j - 1]
                        else:
                            nextPoint = points[j + 1]
                            isNext = True
                        break    
                indexNextPoint = 0            
                for j in range(len(array)):
                    if array[j][0] == nextPoint:
                        indexNextPoint = j 
                        break 
                pointNextPoint = []    
                if isNext: #to the left    
                    dlArrayNextPoint = self.allDl(array, indexNextPoint, -1, -1)
                    self.delRepeatElem(dlArrayNextPoint)
                    for l in range(len(dlArrayNextPoint)): 
                        if self.findNextMinimum(-dlArrayNextPoint[l], array[startIndex:len(array)], array[indexNextPoint], indexNextPoint - startIndex, pointNextPoint, 5):
                            pointNextPoint.append(array[indexNextPoint][0])
                            break 
                else: #to the right
                    dlArrayNextPoint = self.allDl(array, indexNextPoint, len(array), 1)
                    self.delRepeatElem(dlArrayNextPoint)
                    for l in range(len(dlArrayNextPoint)):
                        if self.findNextMinimum(dlArrayNextPoint[l], array[0:startIndex + 1], array[indexNextPoint], indexNextPoint, pointNextPoint, 5):
                            pointNextPoint.append(array[indexNextPoint][0])
                            break
                for j in range(len(pointNextPoint)):
                    if pointNextPoint[j] == array[startIndex][0]: #the start point was found by findMinimum of the next point
                        wasFound = True
                        break
                if wasFound:                 
                   break 
                points.clear()               
        return points 
    def delRepeatElem(self, array):
        array.sort()
        if len(array) > 1:
            for i in range(len(array) - 1, -1, -1):
                if array[i] == array[i - 1]:
                    array.pop(i) 
        return array        
    def allDl(self, array, startIndex, endIndex, step):
        dlArray = []
        for i in range(startIndex + step, endIndex, step):
                dlArray.append(abs(array[startIndex][0] - array[i][0]))
        return dlArray        
    def findNextMinimum(self, dl, valArray, startVal, index, points, errors, maxDistance = 50):
        startPos = startVal[0]
        step = int(dl/abs(dl))      
        availableDistance =  dl * step //5 if abs(dl) < 100 and maxDistance == 50 else maxDistance #dl * step //7
        newPos = startPos + dl
        if newPos < valArray[0][0] - availableDistance or newPos > valArray[len(valArray) - 1][0] + availableDistance or index + step >= len(valArray) or index + step < 0:
            return True
        minDistance = 100
        indexMinDistance = 0
        lastIndex = -1 if dl < 0 else len(valArray)  
        for i in range(index + step, lastIndex, step):
            distance = abs(newPos - valArray[i][0])
            if distance < minDistance:
                minDistance = distance
                indexMinDistance = i
        if minDistance <= availableDistance:
            points.append(valArray[indexMinDistance][0])
            return self.findNextMinimum(dl, valArray, valArray[indexMinDistance], indexMinDistance, points, errors, maxDistance)
        else:
            if errors < 3:
                points.append(startVal[0] + dl)
                return self.findNextMinimum(dl, valArray, (startVal[0] + dl, 0), indexMinDistance, points, errors + 1, 2 * maxDistance)
            else:    
                points.clear()
                return False


'''
This Program is Compactible upto 1 meter approx
And need proper light on face to detect the face and teh distance
'''

import cv2
import mediapipe as mp
import time
from cvzone.FaceMeshModule import FaceMeshDetector

m_detector=FaceMeshDetector(maxFaces=3)

class FaceDetector():
    def __init__(self,minDetectionCon=0.75): 

        self.minDetectionCon=minDetectionCon
        self.mp_face_detection = mp.solutions.face_detection
        self.mp_drawing = mp.solutions.drawing_utils
        self.facedetection=self.mp_face_detection.FaceDetection(0.75)

    def findFaces(self,img,draw=True):

        imgRGB= cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.facedetection.process(imgRGB)
        #print(self.results)
        bboxs = []

        if self.results.detections:
            for id , detection in enumerate(self.results.detections):
                #print(id,detection)
                #print(detection.score)
                #print(detection.location_data.relative_bounding_box)
                #mp_drawing.draw_detection(img,detection)
                

                bboxC=detection.location_data.relative_bounding_box
                ih, iw, ic=img.shape
                bbox=int(bboxC.xmin * iw),int(bboxC.ymin * ih), int(bboxC.width * iw),int(bboxC.height * ih)
                bboxs.append([id, bbox, detection.score])
                
                if draw:
                    img= self.fancyDraw(img,bbox)     

                    self.distance_Finder(img, bbox[0], bbox[1]+bbox[3]+22 )
                    
                    cv2.putText(img, f'{int(detection.score[0]*100)}% Correct', (bbox[0],bbox[1]-20), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
                    
            
        return img, bboxs

    def fancyDraw(self, img ,bbox, l=30, t=5):
        x,y,w,h = bbox
        x1, y1 = x+w, y+h
        l=int((w)/6)
        cv2.rectangle(img,bbox,(100,0,100),3)

        cv2.line(img, (x,y), (x+l,y), (0,255,0), t )
        cv2.line(img, (x,y), (x,y+l), (0,255,0), t )

        cv2.line(img, (x1,y1), (x1-l,y1), (0,255,0), t )
        cv2.line(img, (x1,y1), (x1,y1-l), (0,255,0), t )

        cv2.line(img, (x1-w,y+h), (x1-w+l,y+h), (0,255,0), t )
        cv2.line(img, (x1-w,y+h), (x1-w,y+h-l), (0,255,0), t )

        cv2.line(img, (x+w,y1-h), (x+w-l,y1-h), (0,255,0), t )
        cv2.line(img, (x+w,y1-h), (x+w,y1-h+l), (0,255,0), t )

        return img
    
    def distance_Finder(self,img, a,b):
        imgf, m_faces = m_detector.findFaceMesh(img, draw= False)
        if m_faces:
            for face in m_faces:
                pointLeft_eye = face[145]
                pointRight_eye = face[374]
                
                '''
                cv2.circle(imgf, pointLeft_eye, 5, (0,255,0),cv2.FILLED)
                cv2.circle(imgf, pointRight_eye, 5, (0,255,0),cv2.FILLED)
                cv2.line(imgf, pointLeft_eye, pointRight_eye, (255,0,255), 2)
                '''

                w,_= m_detector.findDistance(pointLeft_eye, pointRight_eye)
                W = 6.3
                #   d= 48
                #   f= (w*d)/W = 600
                f=600
                d=(W*f)/w 

                cv2.putText(img, f'Distance: {int(d)}cm', (a,b ), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)

def main():
    cap=cv2.VideoCapture(0)
    pt=time.time()

    f_detector= FaceDetector()

    while True:
        success, img = cap.read()

        imgr,bbox=f_detector.findFaces(img, draw= True)
        #print(bbox)
        
        ct=time.time()
        fps=(1/(ct-pt))
        pt=ct

        cv2.putText(imgr, f'FPS:{int(fps)}', (20,70), cv2.FONT_HERSHEY_PLAIN, 1, (255,0,0), 1)

        cv2.imshow("Camera:",imgr)

        if cv2.waitKey(1) & 0xFF==27:
            break
    img.release()
    cv2.destroyAllWindows()

if __name__=='__main__':
    main()
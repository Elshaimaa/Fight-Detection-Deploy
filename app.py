#!/usr/bin/env python
# coding: utf-8

# In[1]:
import numpy as np
import os
import base64
import json
from numpy import random
import imutils
import cv2
import pandas as pd
from fight_detection import Fight_utils


# In[5]:



# import the necessary packages
from imutils.video import VideoStream
from flask import Response, Flask, request, redirect, render_template
from flask_cors import CORS, cross_origin
import threading
import argparse
import datetime
import time
import cv2
from pathlib import Path
import torch
import torch.backends.cudnn as cudnn

import sys
sys.path.append('yolov5-crowdhuman')


from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized

from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
from collections import Counter

# In[6]:


# initialize the output frame and a lock used to ensure thread-safe
# exchanges of the output frames (useful when multiple browsers/tabs
# are viewing the stream)
image_path = None
outputFrame = None
output = []
lock = threading.Lock()
# initialize a flask object
app = Flask(__name__)
CORS(app)
# initialize the video stream and allow the camera sensor to
# warmup
#vs = VideoStream(usePiCamera=1).start()

@app.route("/")
@cross_origin()
def root():
    return redirect("/index.html")


def generate_crop():
    global image_path
    im = image_path
    detect(source = image_path, project='facepics', name = image_path[5:])
    directory = 'facepics/'+image_path[5:]
    arr = []
    dic = {}
    # iterate over files in
    # that directory
    index = 0
    for filename in os.listdir(directory):
        dic1 = {}
        if index == 10 :
            break

        index += 1
        image_path = os.path.join(directory, filename)
        outputimage = cv2.imread(image_path,cv2.COLOR_BGR2RGB)
       ######################################




       #####################################
        string = base64.b64encode(cv2.imencode('.jpg', outputimage)[1]).decode()
#         (flag, encodedImage) = cv2.imencode(".jpg", outputimage)
        # jpg_as_text = base64.b64decode(string)
        # print(type(jpg_as_text))


        dic1["id"] = index
        dic1["url"] = "data:image/png;base64," + string
        if filename == im[5:]:
            arr.insert(1 , dic1)
        else:
            arr.append(dic1)
        print("filename", filename ,"image_path[5:]= ",  im[5:])
    dic["toshow"] = arr
    # print(dic)
    json_object = json.dumps(dic)
    return dic

@app.route("/image_feed")
@cross_origin()
def image_feed():
    print("helllo")
    return generate_crop()


@app.route("/detectFaces.html",methods =["GET","POST"])
@cross_origin()
def faces():
    global image_path
    if request.method == "POST":
        print(request.form["action"])
        image_path ="logs/"+ request.form["action"]
        print("we are here")
        return render_template("detectFaces.html")
    else:
        print("we are not here")
        return render_template("detectFaces.html")

def generate_image():
    global image_path
    outputimage = cv2.imread(image_path,cv2.COLOR_BGR2RGB)
    (flag, encodedImage) = cv2.imencode(".jpg", outputimage)
    yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + 
        bytearray(encodedImage) + b'\r\n')

@app.route("/image_feed_original")
@cross_origin()
def image_feed_original():
    return Response(generate_image(),
        mimetype = "multipart/x-mixed-replace; boundary=frame")

@app.route("/logs.html")
@cross_origin()
def logs():
    df = pd.read_csv("logs.csv")
    data1 = """<!DOCTYPE html>
<html>

<head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <meta http-equiv="X-UA-Compatible" content="ie=edge" />
    <title>Simple House - About Page</title>
    <link href="https://fonts.googleapis.com/css?family=Open+Sans:300,400" rel="stylesheet" />
    <link href="../static/css/all.min.css" rel="stylesheet" />
    <link href="../static/css/templatemo-style.css" rel="stylesheet" />
    <script src="https://cdnjs.cloudflare.com/ajax/libs/jquery/2.2.4/jquery.min.js"></script>
    <script src="http://maxcdn.bootstrapcdn.com/bootstrap/3.3.5/js/bootstrap.min.js"></script>
    <link href="http://maxcdn.bootstrapcdn.com/bootstrap/3.3.5/css/bootstrap.min.css" rel="stylesheet"/>
</head>
<!--

Simple House

https://templatemo.com/tm-539-simple-house

-->
<body> 

    <div class="container">
    <!-- Top box -->
        <!-- Logo & Site Name -->
        <div class="placeholder">
            <div class="parallax-window" data-parallax="scroll" data-image-src="../static/img/simple-house-01.jpg">
                <div class="tm-header">
                    <div class="row tm-header-inner">
                        <div class="col-md-6 col-12">
                        
                            <img src="../static/img/simple-house-logo.png" alt="Logo" class="tm-site-logo" /> 
                            <div class="tm-site-text-box">
                                <h1 class="tm-site-title">Logs</h1>
                                <h6 class="tm-site-description">Fight History</h6>  
                            </div>
                        </div>
                        <nav class="col-md-8 col-12 tm-nav">
                            <ul class="tm-nav-ul">
                                <li class="tm-nav-li"><a href="index.html" class="tm-nav-link active">Home</a></li>
                                
                                <li class="tm-nav-li"><a href="logs.html" class="tm-nav-link">Logs</a></li>
                                <li class="tm-nav-li"><a href="contact.html" class="tm-nav-link">Team</a></li>
                            
                            
                            </ul>
                        </nav>  
                    </div>
                </div>
            </div>
        </div>

        <main>
            <header class="row tm-welcome-section">
                <!--<h2 class="col-12 text-center tm-section-title">About Simple House</h2>
                <p class="col-12 text-center">This is about page of simple house template. 
                You can modify and use this HTML template for your website. Total 3 HTML pages included in this template. Header image has a parallax effect.</p>-->
            </header>


    <div id="pricing" class="container-fluid">
                <table class="table table-bordered table-striped text-center">
                <thead>
                   <tr>
                      <th style="text-align: center;" >start_time</th>
                      <th style="text-align: center;" >end_time</th>
                      <th style="text-align: center;" >path</th>
                   </tr>
               </thead><tbody> """
    df = df.iloc[::-1].head(10)
    data2 = ""
    data3 = ""
    for row in range(len(df)):
        data3  = data3 + """<form action="/detectFaces.html" id={} method="post" style="display: none;">
                            <input type="hidden" name="action" value={} />
        </form>""".format(row , df.iloc[row, 2])

        data2 = data2 + """<tr>
                   <td>{}</td>
                   <td>{}</td>

                   <td><li class="tm-nav-li">


                   <a href="javascript:;" onclick="javascript: document.getElementById({})
                   .submit()">{}</a>

                    </li></td>


                   </tr>""".format(df.iloc[row, 0],df.iloc[row, 1],row,df.iloc[row, 2])
        
                       
    data = f"""{data1}{data2}{data3}</tbody></table></main>

        <footer class="tm-footer text-center">
            <p>Copyright &copy; 2021 ITI Fight Detection 
            
            <a rel="nofollow" href="https://github.com/MohamedSebaie/Fight_Detection_From_Surveillance_Cameras-PyTorch_Project">Fight Detection GitHub</a></p>
        </footer>
    </div>
    <script src="../static/js/jquery.min.js"></script>
    <script src="../static/js/parallax.min.js"></script>
</body>
</html>""" 
    print(data)  
     
    return data

@app.route("/contact.html")
@cross_origin()
def contact():
    return render_template("contact.html")


@app.route("/index.html", methods = ["GET","POST"])
@cross_origin()
def index():
    if request.method == "POST":
        seq = int(request.form["seq"])
        path = request.form["path"]
        skip = int(request.form["skip"])
        mode = request.form["modes"]
        
        if mode == 'Streaming':
            t = threading.Thread(target=streaming, args=(seq, path, skip,))
            t.daemon = True
            t.start()
            return redirect("/videoStreaming.html")
        
        elif mode == 'Uploading':
            processing(seq, path, skip)
            return redirect("/videoProcessing.html")
        
        else:
            return render_template("index.html")
        
    else:
        return render_template("index.html")

@app.route("/videoStreaming.html")
@cross_origin()
def videoStreaming():
    # return the rendered template
    return render_template("videoStreaming.html")
def streaming(frameCount,src,skip):
    # grab global references to the video stream, output frame, and
    # lock variables
    global outputFrame, lock
    
    inp = src
    if inp == '0':
        inp = int(inp)

    vs = VideoStream(src=inp).start()
    time.sleep(2.0)
    
    df = pd.read_csv("logs.csv")
    l = []
    state = []
    start = None
    end = None
    last_time = time.time() - 3
    total = 0
    # loop over frames from the video stream
    while True:
        
        # read the next frame from the video stream, resize it,
        # convert the frame to grayscale, and blur it
        frame = vs.read()
        
        if (last_time+2.5 < time.time()) and (total%skip == 0):
            if len(l) == 0:
                start = datetime.datetime.now()
            l.append(frame)
            if len(l) == 16:
                end = datetime.datetime.now()
        print(len(l))
        if len(l) == frameCount:
            last_time = time.time()
#             Fight_utils.streaming_predict(l)
            x = threading.Thread(target=Fight_utils.streaming_predict, args=([l]))
            x.daemon = True
            x.start()
            l = []
            if Fight_utils.predicted_class_name == "fight":
                state.append([1,start,end,frame])
            else:
                state.append([0,start,end,frame])
            #Fight_utils.predicted_class_name = ""
      
        if len(state) >= 4:
            if state[-1][0] == 1 and state[-2][0] == 1 and state[-3][0] == 1 and state[-4][0] == 1:
                path = state[-1][1].strftime("%d-%m-%Y_%H-%M-%S")+'.jpg'
                fpath = ".\\logs\\"+path
                cv2.imwrite(fpath, frame)
                df2 = {'start_time': state[-4][1], 'end_time': state[-1][2], 'path': path}
                df = df.append(df2, ignore_index = True)
                df.to_csv("logs.csv",index=False)
                state = []
        
        predicted_class_name = Fight_utils.predicted_class_name
        if predicted_class_name == "fight":
            cv2.putText(frame, predicted_class_name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
        else:
            cv2.putText(frame, predicted_class_name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        timestamp = datetime.datetime.now()
        cv2.putText(frame, timestamp.strftime(
            "%A %d %B %Y %I:%M:%S%p"), (10, frame.shape[0] - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)
        
        total += 1
        # acquire the lock, set the output frame, and release the
        # lock
        with lock:
            outputFrame = frame.copy()
def generate_feed():
    # grab global references to the output frame and lock variables
    global outputFrame, lock
    # loop over frames from the output stream
    while True:
        # wait until the lock is acquired
        with lock:
            # check if the output frame is available, otherwise skip
            # the iteration of the loop
            if outputFrame is None:
                continue
            # encode the frame in JPEG format
            (flag, encodedImage) = cv2.imencode(".jpg", outputFrame)
            # ensure the frame was successfully encoded
            if not flag:
                continue
        # yield the output frame in the byte format
        yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + 
            bytearray(encodedImage) + b'\r\n')
@app.route("/videoProcessing.html")
@cross_origin()
def videoProcessing():
    # return the rendered template
    return render_template("videoProcessing.html")
def processing(frameCount,src,skip):
    # grab global references to the video stream, output frame, and
    # lock variables
    global outputFrame, lock, output
    
    vr = cv2.VideoCapture(src)
    time.sleep(2.0)
    
    df = pd.read_csv("logs.csv")
    
    l = []
    state = []
    start = None
    end = None
    total = 0
    # loop over frames from the video stream
    while vr.isOpened():
        
        # Read the frame.
        ok, frame = vr.read() 
        
        # Check if frame is not read properly then break the loop.
        if not ok:
            break
        
        if total%skip == 0:
            if len(l) == 0:
                start = datetime.datetime.now()
            l.append(frame)
            if len(l) == 16:
                end = datetime.datetime.now()
        
        print(len(l))
        if len(l) == frameCount:
            Fight_utils.streaming_predict(l)
#             x = threading.Thread(target=Fight_utils.streaming_predict, args=([l]))
#             x.daemon = True
#             x.start()
            l = []
            if Fight_utils.predicted_class_name == "fight":
                state.append([1,start,end,frame])
            else:
                state.append([0,start,end,frame])
#             Fight_utils.predicted_class_name = ""
      
        if len(state) >= 4:
            if state[-1][0] == 1 and state[-2][0] == 1 and state[-3][0] == 1 and state[-4][0] == 1:
                path = state[-1][1].strftime("%d-%m-%Y_%H-%M-%S")+'.jpg'
                fpath = ".\\logs\\"+path
                cv2.imwrite(fpath, frame)
                df2 = {'start_time': state[-4][1], 'end_time': state[-1][2], 'path': path}
                df = df.append(df2, ignore_index = True)
                df.to_csv("logs.csv",index=False)
                state = []
        
        predicted_class_name = Fight_utils.predicted_class_name
        if predicted_class_name == "fight":
            cv2.putText(frame, predicted_class_name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
        else:
            cv2.putText(frame, predicted_class_name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        timestamp = datetime.datetime.now()
        cv2.putText(frame, timestamp.strftime(
            "%A %d %B %Y %I:%M:%S%p"), (10, frame.shape[0] - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)
        
        total += 1
        # acquire the lock, set the output frame, and release the
        # lock
        
        outputFrame = frame.copy()
        output.append(outputFrame)
def generate_upload():
    # grab global references to the output frame and lock variables
    global lock, output
    # loop over frames from the output stream
    for outputFrame in output:
        # check if the output frame is available, otherwise skip
        # the iteration of the loop
        if outputFrame is None:
            continue
        # encode the frame in JPEG format
        (flag, encodedImage) = cv2.imencode(".jpg", outputFrame)
        # ensure the frame was successfully encoded
        if not flag:
            continue
        # yield the output frame in the byte format
        yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + 
            bytearray(encodedImage) + b'\r\n')

@app.route("/video_feed")
@cross_origin()
def video_feed():
    # return the response generated along with the specific media
    # type (mime type)
    return Response(generate_feed(),
        mimetype = "multipart/x-mixed-replace; boundary=frame")
@app.route("/video_upload")
@cross_origin()
def video_upload():
    # return the response generated along with the specific media
    # type (mime type)
    return Response(generate_upload(),
        mimetype = "multipart/x-mixed-replace; boundary=frame")

def detect(save_img=False,agnostic_nms=False, augment=False, classes=None, conf_thres=0.25, device='', exist_ok=False, heads=True, img_size=640, iou_thres=0.45, name='exp', 
           person=False, project='facepics', save_conf=False, save_txt=False, source='logs/25-12-2021_03-10-35.jpg', update=False, view_img=True, 
           weights=['crowdhuman_yolov5m.pt'], eps=0.12):
  
    source, weights, view_img, save_txt, imgsz = source, weights, view_img, save_txt, img_size
    webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
        ('rtsp://', 'rtmp://', 'http://'))
    
    li=[]
    to_draw=[]
    # cropped = []
    face_num = 0  
    image_original = cv2.imread(source)
    #Directories
    save_dir = Path(increment_path(Path(project) / name, exist_ok=exist_ok))  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Initialize
    set_logging()
    device = select_device(device)
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size
    if half:
        model.half()  # to FP16

    # Second-stage classifier
    classify = False
    if classify:
        modelc = load_classifier(name='resnet101', n=2)  # initialize
        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model']).to(device).eval()

    # Set Dataloader
    vid_path, vid_writer = None, None
    if webcam:
        view_img = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride)
    else:
        save_img = True
        dataset = LoadImages(source, img_size=imgsz, stride=stride)

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    t0 = time.time()
    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        t1 = time_synchronized()
        pred = model(img, augment=augment)[0]
        # print(f'prediction ya sba3y aho \n\n\n {pred.shape} \n\n\n')

        # Apply NMS
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes=classes, agnostic=agnostic_nms)
        t2 = time_synchronized()

        # Apply Classifier
        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            if webcam:  # batch_size >= 1
                p, s, im0, frame = path[i], '%g: ' % i, im0s[i].copy(), dataset.count
            else:
                p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # img.jpg
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # img.txt
            s += '%gx%g ' % img.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results
                id = -1
                for *xyxy, conf, cls in reversed(det):
                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist() # normalized xywh                        
                        line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                        with open(txt_path + '.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    if save_img or view_img:  # Add bbox to image
                        label = f'{names[int(cls)]} {conf:.2f}'
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist() # normalized xywh
                        
                        # print(type(xywh))
                        if heads or person:
                            if 'head' in label and heads:
                                dimention=xywh
                                
                                li.append(dimention)
                                to_draw.append(xyxy)
                                #print(dimention.type)
                                #print(f'\n \n {xywh[:2]}  \n \n')
                                # print(f' bos hena \n \n  \n  {xyxy} \n \n')
                                
                                                                
                                # plot_one_box(xyxy, im0, label=f'{id}', color=colors[int(cls)], line_thickness=3)

                                #############
                                # id += 1
                                # label = f'{id}'
                                # tl=1
                                # # color = color or [random.randint(0, 255) for _ in range(3)]
                                # color = colors[int(cls)] or [random.randint(0, 255) for _ in range(3)]
                                # c1, c2 = (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3]))
                                # cv2.rectangle(im0, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)

                                # tf = max(tl - 1, 1)  # font thickness
                                # t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
                                # c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
                                # cv2.rectangle(im0, c1, c2, color, -1, cv2.LINE_AA)  # filled
                                # cv2.putText(im0, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)   
                                #############



                            if 'person' in label and person:
                                plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=3)
                        else:
                            plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=3)

            # Print time (inference + NMS)
            print(f'{s}Done. ({t2 - t1:.3f}s)')

            # Stream results
            if view_img:
                # cv2.imshow(str(p), im0)
                cv2.waitKey(0)  # 1 millisecond

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'image':

                    #######################
                    colors = [(255,0,0), (0,255,0), (0,0,255), (255,255,0), (0,255,255), (255,0,255), (192,192,192), (128,128,128), (128,0,0), (128,128,0), (0,128,0), (128,0,128), (0,128,128), (255,255,255)]


                    matrix=np.asarray(li)                 
                    db1 = DBSCAN(eps = eps, min_samples = 2).fit(matrix)
                    classes = db1.labels_
                    counting_classes = Counter(classes)                    
                    list_tuples = counting_classes.most_common(1)  # <---- 3'aairo el rakam da 3ashan yetla3 el sowar men kam cluster
                    max_len = [i[0] for i in list_tuples]
                    # classes = classes + 1 if -1 in classes else classes                    
                    d1 = {index : value for index, value in enumerate(classes)}              
                    
                    for index, xyxy in enumerate(to_draw): 
                      color = colors[d1[index]]

                      tl=1
                      c1, c2 = (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3]))

                      
                      y1 = int(xyxy[1])
                      y2 = int(xyxy[3])
                      x1 = int(xyxy[0])
                      x2 = int(xyxy[2])
                      

                      crop = image_original[y1:y2,x1:x2]

                      # save crop                                           
                      cropped_path = str(save_dir) + f'/{face_num}.png'
                      if d1[face_num] in max_len:
                        cv2.imwrite(cropped_path, crop)
                      face_num += 1
                      # save crop

                      # cropped.append(crop)

                      cv2.rectangle(im0, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
                      tf = max(tl - 1, 1)  # font thickness
                      t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
                      c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
                      cv2.rectangle(im0, c1, c2, [255,255,255], -1, cv2.LINE_AA)  # filled
                      cv2.putText(im0, str(d1[index]), (c1[0], c1[1] - 2), 0, tl / 3, [0, 0, 0], thickness=tf, lineType=cv2.LINE_AA) 
                   
                    #######################

                    # plt.imshow(cropped[4])
                    # cropped_path = str(save_dir) + '/1.png'
                    # print(f'\n el esm howa {cropped_path} \n')
                    # print(f'\n el esm howa {save_path} \n') 
                    ###########################################################
                    # scale_percent = 60# percent of original size
                    # width = int(im0.shape[1] * scale_percent / 100)
                    # height = int(im0.shape[0] * scale_percent / 100)
                    # dim = (width, height)
                      
                    # # resize image
                    # im0 = cv2.resize(im0, dim, interpolation = cv2.INTER_AREA)





                    ############################################################


                    cv2.imwrite(save_path, im0)                   
                    

                # else:  # 'video'
                #     if vid_path != save_path:  # new video
                #         vid_path = save_path
                #         if isinstance(vid_writer, cv2.VideoWriter):
                #             vid_writer.release()  # release previous video writer

                #         fourcc = 'mp4v'  # output video codec
                #         fps = vid_cap.get(cv2.CAP_PROP_FPS)
                #         w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                #         h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                #         vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*fourcc), fps, (w, h))
                #     vid_writer.write(im0)

    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        print(f"Results saved to {save_dir}{s}")
    matrix=np.asarray(li)
    #print(matrix)
    print(f'Done. ({time.time() - t0:.3f}s)')
    return matrix

if __name__ == '__main__':
    # construct the argument parser and parse command line arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--ip", type=str, required=True,
        help="ip address of the device")
    ap.add_argument("-o", "--port", type=int, required=True,
        help="ephemeral port number of the server (1024 to 65535)")
    
    args = vars(ap.parse_args())
        # start the flask app
    app.run(host=args["ip"], port=args["port"], debug=True,
        threaded=True, use_reloader=False)
# release the video stream pointer
vs.stop()


# In[ ]:





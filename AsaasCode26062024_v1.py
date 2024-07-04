import cv2
import os
import docker
import json
import time
import stat
import numpy as np
import yaml
import math
import boto3
from botocore.exceptions import NoCredentialsError, PartialCredentialsError
import requests
from flask import Flask, request
app = Flask(__name__)

#Function to download the video file
# def download_video(url, local_filename):
#     # Streaming download
#     with requests.get(url, stream=True) as r:
#         r.raise_for_status()
#         with open(local_filename, 'wb') as f:
#             for chunk in r.iter_content(chunk_size=8192):
#                 f.write(chunk)

def splitVideoFrames(url):
    '''
    This function is the first step of the process - To split an incoming video into its frames,
    in an images folder inside a parent folder of the project.
    '''

    # Remove trailing slash if present and split by '/'
    path_segments = url.rstrip('/').split('/')

    # Get the last segment
    user_ID, video_name = path_segments[-2:]

    
    #Get Video  from URL
    #download_video(url, video_file)
    base_directory = os.getcwd()
    video_file = base_directory + '/' + user_ID + '/' + video_name
    projectFolder, _ = os.path.splitext(video_file)
    projectFolder += "-project"
    imagesFolder = f"{projectFolder}/images"

    print(f"Project Folder: {projectFolder}")
    print(f"Base Folder: {base_directory}")
  
    # projectFolder, _ = os.path.splitext(video_file)
    # projectFolder += "-project"
    # imagesFolder = f"{projectFolder}/images"
   
    # make a parent folder by the name of the video file
    if not os.path.isdir(projectFolder):
        os.mkdir(projectFolder)
        # make an images folder in parent directory
        os.mkdir(imagesFolder)
   
    # make a parent folder by the name of the video file
    if not os.path.isdir(projectFolder):
        os.mkdir(projectFolder)
        os.chmod(projectFolder, stat.S_IRWU | stat.S_IRGRP | stat.S_IROTH )

        # make an images folder in parent directory
        os.mkdir(imagesFolder)
        os.chmod(imagesFolder, stat.S_IRWU | stat.S_IRGRP | stat.S_IROTH )
        
    # read the video file   
    #if (url)

    full_URL  = "https://bm-3drecon-prod.s3.amazonaws.com/" + url
    cap = cv2.VideoCapture(full_URL)
    # start the loop
    imgsCounter = 0 
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    totalFramesLength = len(str(total_frames)) #Get length of total frames as string
    while True:
        imgsCounterLength = len(str(imgsCounter)) #Get length of current counter as string
        
        # Give each image a name of equal string length
        if imgsCounterLength != totalFramesLength:
            imgName = "img_" + "0"*(totalFramesLength-imgsCounterLength) +f"{imgsCounter}.jpg"
        else:
            imgName = f"img_{imgsCounter}.jpg"
        
        is_read, frame = cap.read()
        if not is_read:
            # break out of the loop if there are no frames to read
            break
        #Check if image exists
        elif os.path.isfile(os.path.join(imagesFolder,imgName)): 
            print(f"image {imgName} skipped")
            imgsCounter += 1
            continue
        else:
            cv2.imwrite(os.path.join(imagesFolder, imgName), frame) 
            print(f"image {imgName} created")
            # increment the frame count
            imgsCounter += 1
        
    # else:
    #     print("Video file does not exist. Skipping this step.")
    #     imgsCounter = sum(os.path.isfile(os.path.join(imagesFolder, f)) for f in os.listdir(imagesFolder))
    #     print(f"Found {imgsCounter} Images.")
    
    cam_models_overrides_file(projectFolder) #create JSON file
    config_file(projectFolder, imgsCounter) #create Yaml file
    return projectFolder, imagesFolder

def cam_models_overrides_file (projectFolder):
    '''
    Creates camera models override json file in parent folder.
    '''
    config = {
        "all": {
            "width": 3840, 
            "projection_type": "equirectangular",
            "height": 1920
        }
    }

    try:
        with open(os.path.join(projectFolder,'camera_models_overrides.json'), 'w') as outfile:
            json.dump(config, outfile)
        print("Created camera_models_overrides.json succesfully.")

    except Exception as e:
        print("Error creating JSON file.")
        print(e)

def config_file (projectFolder, imgsCount):
    '''
    Creates config YAML file in parent folder.
    '''
    processes = 8  #fixed
    config = {
        "processes": processes, # Number of threads to use
        "feature_process_size" : 2048,    # Resize the image if its size is larger than specified. Set to -1 for original size
        "feature_min_frames_panoramas": 16000,    # If fewer frames are detected, sift_peak_threshold/surf_hessian_threshold is reduced.
        "matching_gps_neighbors" : 4,     # Number of images to match selected by GPS distance. Set to 0 to use no limit
        "feature_process_size_panorama" : 2048 #Resize Images
    }
    if (os.path.isfile(os.path.join(projectFolder,'config.yaml'))):
        print("config.yaml file exists. Skipping step.")
    else:
        try:
            with open(os.path.join(projectFolder,'config.yaml'), 'w') as outfile:
                yaml.dump(config, outfile)
            print("Created Config.yaml succesfully.")
            
        except Exception as e:
            print("Error creating YAML file.")
            print(e)

def startDocker (projectFolder):
    '''
    This function is to start a docker container from an image, and pass a command to it [openSFM]
    Params:
    projectFolder: Folder containing project's files
    '''
    if (os.path.isfile(os.path.join(projectFolder,'reconstruction.json'))):
        print("Reconstruction file exists. Skipping docker container step.")
    else:
        image = "opensfm_bm" #image name goes here - custom image: opensfm_bm
        projectName = os.path.basename(projectFolder)
        command = f"bin/opensfm_run_BM /{projectName}" #bin/opensfm_run_BM
                
        client = docker.from_env()
        bindVolume = {
            f'{projectFolder}': {
                'bind': f'/{projectName}', #Change datasets to dynamic variable for project name
                'mode': 'rw'
            }
        }
        container = client.containers.run(image, command, detach=True, volumes=bindVolume)
        
        #Wait for the container to finish
        print("Docker Container Started. Waiting for it to finish. This could take a while.")
        container.wait()

def normalize_coordinates(coordinates):
    '''Normalization of x, y, coordinates from Opensfm JSON results
    Params:
    coordinates: array of coordinates'''
    # Find the minimum and maximum values for each dimension
    min_vals = np.min(coordinates, axis=0)
    max_vals = np.max(coordinates, axis=0)

    # Normalize each dimension to the range [0, 1]
    normalized_coordinates = (coordinates - min_vals) / (max_vals - min_vals)

    return normalized_coordinates

def rotation_matrix_from_Rot_values(vector):
    # Compute the length (magnitude) of the vector
    length = np.linalg.norm(vector)

    # Compute the direction (unit vector) of the vector
    direction = vector / length if length != 0 else vector

    # Compute components for the rotation matrix
    cos_theta = np.cos(length)
    sin_theta = np.sin(length)
    one_minus_cos_theta = 1 - cos_theta

    # Construct the rotation matrix
    rotation_matrix = cos_theta * np.eye(3) + \
                      one_minus_cos_theta * np.outer(direction, direction) + \
                      sin_theta * np.array([[0, -direction[2], direction[1]],
                                            [direction[2], 0, -direction[0]],
                                            [-direction[1], direction[0], 0]])
    
    return rotation_matrix

def calculate_direction(point1, point2):
    # Calculate the angle of the line formed by two points with respect to the positive x-axis
    dx = float(point2[0]) - float(point1[0])
    dy = float(point2[1]) - float(point1[1])
    angle_rad = math.atan2(dy, dx)
    angle_deg = math.degrees(angle_rad)
    
    # Ensure the angle is between 0 and 360 degrees
    if angle_deg < 0:
        angle_deg += 360
    
    return angle_deg

def plot_equirectangular_cameras(shots):

    camera_positions = []
    # cameras3D = []
    for i, (shot_name, shot_data) in enumerate(sorted(shots.items())):
       
        # Extract rotation and translation matrices
        rotation_matrix = rotation_matrix_from_Rot_values(np.array(shot_data["rotation"])) #rotation_angles = np.array(shot_data["rotation"])
        translation_vector = shot_data["translation"]
       
        # Find camera location
        rot_camera = np.dot(rotation_matrix, translation_vector)
        
        camera_positions.append((-rot_camera[0],rot_camera[1])) #Was 1 then 0 but changed to (top, left)
    
    #Normalize Camera 2D Positions     
    norm_cam_positions = normalize_coordinates(camera_positions).tolist()

    return norm_cam_positions

def processReconstructionFile(user_ID, project_ID, projectFolder, imagesFolder):
    '''
    This function is to process the reconstruction JSON file once it has been created.
    '''
    
    #open Json File once it exists
    with open(os.path.join(projectFolder,'reconstruction.json')) as f_in:
        data = json.load(f_in)

    outputFile = {

    }
    projectName = os.path.basename(projectFolder)
    for i, cluster in enumerate(data): 

        #For Each Cluster do the following:
        #Get Camera Info
        camera = cluster["cameras"].items()
        for key, value in camera:
            frameheight = value["height"]
            framewidth = value["width"]
            frameCenter = ((framewidth/2), (frameheight/2))

        #Get list of image names
        imgs = cluster["shots"].items()
        imgNames = []
        for key, value in imgs:
            imgNames.append(key)


        #Get coordinates for shots
        shots = cluster["shots"]
        cams = plot_equirectangular_cameras(shots)
        
        outputFile[i] = {val: coord for val, coord in zip(sorted(imgNames),cams)} #imgNames

    #Success Code
    if(outputFile):
        successCode = 1
    else:
        successCode = 2

    #Add project info to json
    outputFile['info'] = {
        'User ID':user_ID,
        'Project ID': project_ID,
        'Success Code': successCode, # (1 = success, 2 failure)
        'Images Paths': imagesFolder,
        'Reconstruction File': f"{projectFolder}/reconstruction.json"
    }

    #Create json file  - is this file correctly sorted? each image to its corresponding coords? to be checked   
    try:
        fileName = f'{projectName}_coordinates.json'
        
        with open(os.path.join(projectFolder,fileName), 'w') as outfile:
            json.dump(outputFile, outfile)
        print(f"Created {fileName} successfully.")
    except Exception as e:
        print("Error creating JSON file.")
        print(e)
    return cams, fileName

def rotate_point(point, d_calc, d_chosen):
    
    # Convert angle from degrees to radians
    d_calc = math.radians(d_calc)
    d_chosen = math.radians(d_chosen)

    # Calculate the angle difference between line 1 and line 2
    difference = abs(d_calc - d_chosen)
    if difference > 180:
        difference = 360 - difference
    
    # Unpack the point coordinates
    x, y = point
    
    # Calculate new coordinates after rotation
    x_rotated = x * math.cos(difference) - y * math.sin(difference)
    y_rotated = x * math.sin(difference) + y * math.cos(difference)
    
    return x_rotated, y_rotated

def read_image_from_url(url):
    # Fetch the image from the URL
    response = requests.get(url)
    response.raise_for_status()  # Raise an exception for HTTP errors
    
    # Convert the image to a NumPy array
    image_array = np.frombuffer(response.content, np.uint8)
    
    # Decode the image array into an OpenCV image
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    
    return image  




########### for OpenSFM processing ###########
@app.route('/run', methods=['GET', 'POST'], endpoint="run_opensfm")
def run_opensfm():


    # Check if the request contains JSON data
    if request.is_json:
        # Extract arguments from JSON payload
        data = request.get_json()
        arg1 = data.get('ID')
        arg2 = data.get('vid')
        arg3 = data.get('P')
       
    else:
        # If not JSON, try to extract arguments from URL parameters
        arg1 = request.args.get('ID')
        arg2 = request.args.get('vid') #URL
        arg3 = request.args.get('P')

    # Check if both arguments are provided
    if arg1 is not None and arg2 is not None and arg3 is not None:
        ######### MAIN #############
        #Send success response
        successUrl = "http://benaatech.com/preresponse.php"
        message = f'JSON file received successfully with params: {arg1}, {arg2}, {arg3}'
    
        try:
            # Send the POST request with JSON data
            response = requests.post(successUrl, data=message)
            # Check if the request was successful
            if response.status_code == 200:
                print('Success:', response.text)
            else:
                print('Failed:', response.status_code, response.text)
        except requests.exceptions.RequestException as e:
            # Print the exception if any error occurs
            print('Error:', e)
        
        start_time = time.time()
        #Step 1
        projectFolder, imagesFolder = splitVideoFrames(arg2)
        #Step 2
        startDocker(projectFolder)
        #Step 3
        cams, fileName = processReconstructionFile(arg1, arg3, projectFolder, imagesFolder)
        
        print("Running Time: ", round(time.time() - start_time), "second(s)") #Prints total time of running
        
        #URL to send response to
        url = "http://benaatech.com/response.php"

        # Convert the data to JSON
        headers = {'Content-Type': 'application/json'}
        
        # Read the JSON file
        with open(os.path.join(projectFolder,fileName), 'r') as file:
            data = json.load(file)

        response = requests.post(url, data=json.dumps(data), headers=headers)
        
        return 'Task Completed Succesfully.', 200 #send_file(os.path.join(projectFolder,fileName)), 200 #send_file must be returned.
    else:
        return 'All arguments are required.', 400


########### for projecting points on plans ###########
@app.route('/project', methods=['POST'],  endpoint="run_script")
def run_script():
  
    # Check if the request contains JSON data
    if request.is_json:
        # Extract arguments from JSON payload
        data = request.get_json()
        normalized_points = data.get('point')
        info = data.get('info')
        canvas = read_image_from_url(info.get('map_img')) #link


        #Define Start/ End points
        start_point = data.get('start_point')[:2]
        start_point = (float(start_point[0]),float(start_point[1]))
        end_point = data.get('end_point')[:2]
        end_point = (float(end_point[0]),float(end_point[1]))
        #get Image H, W
     
        h, w, c = canvas.shape

        #scale down image to max w= 300
        scaleDown = w/300
        h = h/scaleDown
        w = w/scaleDown

        #calculate distance between user points
        dx = float(end_point[0]) - float(start_point[0])
        dy = float(end_point[1]) - float(start_point[1])
        distance_for_chosen_points = math.sqrt((dx )**2 + (dy )**2)
        direction_chosen_line = calculate_direction(start_point, end_point)

        #calculate distance between first and last cam
        numOfFrames = len(normalized_points) -1
        dx_cams = float(normalized_points[numOfFrames][0]) - float(normalized_points[0][0])
        dy_cams = float(normalized_points[numOfFrames][1]) - float(normalized_points[0][1])
        distance_between_cams = math.sqrt((dx_cams )**2 + (dy_cams )**2)
        pointsDirection = calculate_direction(normalized_points[0], normalized_points[numOfFrames])
        
        #scaling factor
        scaling_factor = float(distance_for_chosen_points/distance_between_cams)
        
        new_points =  {}
        startPtLarger = False
        xStartLarger = False
        yStartLarger = False

        for i, cam in enumerate(normalized_points):

            pt = (float((cam[0])), float((cam[1])))
            imgName = cam[2]

            scale_pt = np.multiply(pt, scaling_factor)

            #rotate point
            rot_pt = rotate_point(scale_pt, pointsDirection, direction_chosen_line)
            # Translate the coordinates to the origin
            if i == 0:
                difference = abs(np.subtract(rot_pt,start_point)) 
                
                #Logic gates to return correct coordinates
                if start_point[0] > rot_pt[0] and start_point[1] > rot_pt[1]: #If X, Y of start point is larger than resulting rot point
                    startPtLarger = True
                elif start_point[0] < rot_pt[0] and start_point[1] < rot_pt[1]: #If X, Y of start point is smaller than resulting rot point
                    startPtLarger = False
                elif start_point[0] > rot_pt[0] and start_point[1] < rot_pt[1]: #If X only of start point is larger than resulting X of rot point 
                    xStartLarger = True
                    yStartLarger = False
                elif start_point[0] < rot_pt[0] and start_point[1] > rot_pt[1]: #If Y only of start point is larger than resulting Y of rot point 
                    xStartLarger = False
                    yStartLarger = True
            
            #Do math accordingly.
            if startPtLarger:
                trans_pt = (np.add(difference,rot_pt)) 
            elif xStartLarger:
                trans_pt = ((difference[0]+rot_pt[0]),(rot_pt[1]-difference[1])) 
            elif yStartLarger:
                trans_pt = ((rot_pt[0]-difference[0]),(rot_pt[1]+difference[1]))
            else:
                trans_pt = (np.subtract(rot_pt,difference))  

            #Filter points if out of range of image size
            if trans_pt[0] < 0 or trans_pt[1] < 0 or trans_pt[0] > h or trans_pt[1] > w:
                new_points[f"{imgName}"] = 0 
            else:
                trans_pt = (int(trans_pt[0]), int(trans_pt[1]))
                new_points[f"{imgName}"] = trans_pt

        outputFile = {
            "point": new_points,
            "info" : info
        }
        return outputFile, 200

    else:
        return 'JSON is required.', 400

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)  # Run the Flask app in debug mode



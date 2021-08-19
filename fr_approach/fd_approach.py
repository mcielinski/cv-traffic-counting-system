import os
import re
import sys
from os.path import isfile, join

import cv2
import numpy as np
from collections import defaultdict
from scipy import spatial
from tqdm import tqdm


# binarization
THRESHHOLD = 30
# dilatation
KERNEL_SIZE = 3
DILATE_ITER = 5


def vidcap_describe(video_width, video_height, video_fps, video_frames_count):
    print('========= VIDEO DESCRIPTION =========')
    print(f'Width: \t\t{round(video_width)}')
    print(f'Height: \t{round(video_height)}')
    print(f'FPS: \t\t{round(video_fps)}')
    print(f'Frames count:\t{round(video_frames_count)}')
    print('=====================================')


# PURPOSE: Draw detection zone 
# PARAMETERS: 
#   - detection_zone - detection zone placement
#   - frame - frame on which the detection zone is displayed
def draw_detection_zone(detection_zone, frame):
    cv2.rectangle(
        frame, 
        detection_zone[0],
        detection_zone[1], 
        (255,1,1), 
        3
    )


# PURPOSE: Display the vehicle count on the top-left corner of the frame
# PARAMETERS: 
#   - frame - frame on which the count is displayed
#   - vehicle_count - number of vehicles
def display_vehicle_count(vehicle_count, frame):
    cv2.putText(
        frame,
        'Detected Vehicles: ' + str(vehicle_count), 
        (20, 45), 
        cv2.FONT_HERSHEY_SIMPLEX, 
        0.8, 
        (0, 0xFF, 0), 
        2
    )


# PURPOSE: Draw detection box id on its center
# PARAMETERS: 
#   - current_box_id - current box id
#   - current box center - current box center
#   - frame - frame on which the detection boxes are displayed
def draw_detection_box_id(current_box_id, current_box_center, frame):
    cv2.putText(
        frame, 
        current_box_id, 
        current_box_center, 
        cv2.FONT_HERSHEY_SIMPLEX, 
        0.5, 
        [0, 0, 255], 
        2
    )


# PURPOSE: Identifying if the current box was present in the previous frames
# PARAMETERS: 
#   - prev_frames_detections - all the vehicular detections of N previous frames (N = FRAMES_BEFORE_CURRENT)
#   - current_box - current box
#   - current_detections - the coordinates of the box of previous detections
#   - frames_defore_current - number of frames saved in history
# RETURN: 
#   - False - if the box was present in previous frames
#	- True - if the box wasn't present in previous frames (is new)
def is_new_object(prev_frames_detections, current_box, current_detections, frames_defore_current):
    center_x, center_y, width, height = current_box

    dist = np.inf
    # Iterating through all the k-dimensional trees
    for i in range(frames_defore_current):
        prev_frame_detections = list(prev_frames_detections[i].keys())
        if len(prev_frame_detections) == 0:
            continue
        # Finding the distance to the closest point and the index
        temp_dist, index = spatial.KDTree(prev_frame_detections).query([(center_x, center_y)])
        if temp_dist < dist:
            dist = temp_dist
            frame_num = i
            coords = prev_frame_detections[index[0]]

    if dist > (max(width, height) / 2):
        return True

    # Keeping the vehicle ID constant
    current_detections[(center_x, center_y)] = prev_frames_detections[frame_num][coords]

    return False


# PURPOSE: Get current vehicle count + valid detections
# PARAMETERS:
#   - boxes - all bounding boxes info in detection zone (e.g. [top_left_x, top_left_y, width, height, center_x, center_y])
#   - vehicle_count - current vehicle count
#   - prev_frames_detections - all the vehicular detections of N previous frames (N = FRAMES_BEFORE_CURRENT)
#   - frames_defore_current - number of frames saved in history
# RETURN: 
#   - vehicle_count - new vehicle count
#	- current_detections - current detections
def count_vehicles(boxes, vehicle_count, prev_frames_detections, frames_defore_current):
    current_detections = {}

    for box in boxes:
        w = box[2]
        h = box[3]
        center_x = box[4]
        center_y = box[5]
    
        current_detections[(center_x, center_y)] = vehicle_count
        current_box = (center_x, center_y, w, h)

        # When the id of the detection is not present in previous frames detections
        if is_new_object(prev_frames_detections, current_box, current_detections, frames_defore_current):
            vehicle_count += 1

        # Get the id corresponding to the current detection
        current_box_id = current_detections[(center_x, center_y)]

        # When two detections have the same id (due to being too close)
        if list(current_detections.values()).count(current_box_id) > 1:
            current_detections[(center_x, center_y)] = vehicle_count
            vehicle_count += 1
        
        # Get the id corresponding to the current detection
        current_box_id = current_detections[(center_x, center_y)]

        # Display the id at the center of the box
        current_box_center = (center_x, center_y)

    return vehicle_count, current_detections


# PURPOSE: Initializing the video writer with the output video path and the same number
#   of fps, width and height as the source video
# PARAMETERS: 
#   - video_width - video width
#   - video_height - video height
#   - video_fps - video FPS
#   - out_path - output video path
# RETURN: The initialized video writer
def initialize_video_writer(video_width, video_height, video_fps, out_path):
    video_writer = cv2.VideoWriter(
        out_path, 
        cv2.VideoWriter_fourcc(*'XVID'), # XVID, MJPG, DIVX
        video_fps, 
        (video_width, video_height))

    return video_writer


def main(argv):
    if len(argv) < 4 or '--input_path' not in argv or '--output_path' not in argv:
        print(
            '\n================================================================ \n'
            'USAGE:\n'
            '  python fd_approach.py --input_path <path_to_video_file> --output_path <output_video_dir>\n'
            'OPTIONS: \n'
            '  Detection zone width percentage (calculated from the bottom of the image) \n'
            '    --dz_height_percentage <e.g. 0.8> \n'
            '  Detection zone width percentage (calculated form the left side of the image) \n'
            '    --dz_width_percentage <e.g. 0.4> \n'
            '  Minimum contour size, that will be detected \n'
            '    --contour_area <area> \n'
            '================================================================\n'
        )
        sys.exit()

    try:
        in_path_argv = argv.index('--input_path')
        video_in_path = argv[in_path_argv+1]
        video_file_name = video_in_path.split('/')[-1].split('.')[0]
        video_file = video_file_name + '.avi'

        out_path_argv = argv.index('--output_path')
        video_out_dir = argv[out_path_argv+1]
        video_out_path = os.path.join(video_out_dir, video_file)

        if not os.path.exists(video_out_dir):
            os.makedirs(video_out_dir)
    except IndexError:
        print('Invalid input!')
        sys.exit()

    try:
        dz_height_percentage_argv = argv.index('--dz_height_percentage')
        dz_height_percentage = argv[dz_height_percentage_argv+1]
        # if not dz_height_percentage.isdigit():
        #     print('Invalid input!')
        #     sys.exit()
        DETECTION_ZONE_H_PERCENTAGE = float(dz_height_percentage)
    except ValueError:
        DETECTION_ZONE_H_PERCENTAGE = 0.4
    except IndexError:
        print('Invalid input!')
        sys.exit()
    
    try:
        dz_width_percentage_argv = argv.index('--dz_width_percentage')
        dz_width_percentage = argv[dz_width_percentage_argv+1]
        # if not dz_width_percentage.isdigit():
        #     print('Invalid input!')
        #     sys.exit()
        DETECTION_ZONE_W_PERCENTAGE = float(dz_width_percentage)
    except ValueError:
        DETECTION_ZONE_W_PERCENTAGE = 1.0
    except IndexError:
        print('Invalid input!')
        sys.exit()
    
    try:
        contour_area_argv = argv.index('--contour_area')
        contour_area = argv[contour_area_argv+1]
        if not contour_area.isdigit():
            print('Invalid input!')
            sys.exit()
        COUNTOUR_AREA = int(contour_area)
    except ValueError:
        COUNTOUR_AREA = 1000
    except IndexError:
        print('Invalid input!')
        sys.exit()

    # print(video_in_path)
    # print(video_out_path)
    # print(DETECTION_ZONE_H_PERCENTAGE)
    # print(DETECTION_ZONE_W_PERCENTAGE)
    # print(COUNTOUR_AREA)

    # Initialize the video stream pointer to output video file
    video_cap = cv2.VideoCapture(video_in_path)

    video_width = int(video_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    video_height = int(video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video_fps = int(video_cap.get(cv2.CAP_PROP_FPS))
    video_frames_count = int(video_cap.get(cv2.CAP_PROP_FRAME_COUNT))

    writer = initialize_video_writer(video_width, video_height, video_fps, video_out_path)

    vidcap_describe(
        video_width,
        video_height,
        video_fps,
        video_frames_count
    )

    FRAMES_BEFORE_CURRENT = round(video_fps / 5)  # (e.g. 60/4 = 15)

    detection_zone_x1 = 0
    detection_zone_y1 = int(video_height * (1-DETECTION_ZONE_H_PERCENTAGE))
    detection_zone_x2 = int(video_width * DETECTION_ZONE_W_PERCENTAGE)
    detection_zone_y2 = video_height
    DETECTION_ZONE = [
        (detection_zone_x1, detection_zone_y1),
        (detection_zone_x2, detection_zone_y2)
    ]


    # Initialization
    prev_frames_detections = [{(0, 0): 0} for i in range(FRAMES_BEFORE_CURRENT)]
    vehicle_count = 0
    frames_valid_bboxes = []
    frames_detections = []
    detection_occurrences = {}
    detection_occurrences = defaultdict(lambda: 0, detection_occurrences)
    video_cap_images = []

    (grabbed, frame) = video_cap.read()

    if not grabbed:
        print('This video file does not contain a single frame.')
        sys.exit()
    video_cap_images.append(frame)

    prev_frame = frame

    # kernel for image dilation
    kernel = np.ones((KERNEL_SIZE, KERNEL_SIZE),np.uint8)

    print('Pre-processing...')
    for _ in tqdm(range(video_frames_count-1)):
        # Read the next video frame
        (grabbed, frame) = video_cap.read()

        # If the frame was not grabbed, then we have reached the end of the stream
        if not grabbed:
            break
        video_cap_images.append(frame)

        # frame differencing
        grayA = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        grayB = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        diff_image = cv2.absdiff(grayB, grayA)
        
        # image thresholding
        thresh = cv2.threshold(diff_image, THRESHHOLD, 255, cv2.THRESH_BINARY)[1]
        # thresh = cv2.adaptiveThreshold(diff_image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, -20)
        
        # image dilation
        dilated = cv2.dilate(thresh, kernel, iterations=DILATE_ITER)
        
        # find contours
        contours, hierarchy = cv2.findContours(dilated.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        
        # shortlist contours appearing in the detection zone
        valid_bboxes = [[], []]

        for cntr in contours:
            x, y, w, h = cv2.boundingRect(cntr)
            x1, x2 = x, x+w
            y1, y2 = y, y+h
            center_x, center_y = int(x+(w/2)), int(y+(h/2))

            cntr_rect = np.asarray([
                [[x1, y1]], 
                [[x2, y1]],
                [[x2, y2]],
                [[x1, y2]]
            ])

            if (x >= DETECTION_ZONE[0][0]) and \
            (x <= DETECTION_ZONE[1][0]) and \
            (y >= DETECTION_ZONE[0][1]) and \
            (center_y < DETECTION_ZONE[1][1]) and \
            (cv2.contourArea(cntr_rect) >= COUNTOUR_AREA):
                # print(cv2.contourArea(cntr_rect))
                valid_bboxes[0].append(
                    cntr_rect
                )
                valid_bboxes[1].append(
                    [x, y, w, h, center_x, center_y]
                )

        vehicle_count, current_detections = count_vehicles(
            boxes=valid_bboxes[1], 
            vehicle_count=vehicle_count, 
            prev_frames_detections=prev_frames_detections,
            frames_defore_current=FRAMES_BEFORE_CURRENT
        )

        frames_valid_bboxes.append(valid_bboxes)
        frames_detections.append(current_detections)
        for cd_value in list(current_detections.values()):
            detection_occurrences[cd_value] +=1

        # Updating prev_frames_detections
        # Removing the first frame from the list
        prev_frames_detections.pop(0)
        prev_frames_detections.append(current_detections)

        prev_frame = frame

    # ---
    detection_occurrences = dict(detection_occurrences)
    valid_detection_ids_map = {}
    new_detection_id = 0
    for k, v in detection_occurrences.items():
        if v > 3:
            new_detection_id += 1
            valid_detection_ids_map[k] = new_detection_id
    
    # ---
    valid_ids = list(valid_detection_ids_map.keys())
    valid_frames_detections = []
    frames_vehicle_count = []
    vehicle_count = 0
    for frame_detections in frames_detections:
        valid_frame_detections = {}
        for fd_center_point, fr_id in frame_detections.items():
            if fr_id in valid_ids:
                valid_frame_detections[fd_center_point] = valid_detection_ids_map[fr_id]
                if valid_frame_detections[fd_center_point] > vehicle_count:
                    vehicle_count = valid_frame_detections[fd_center_point]

        valid_frames_detections.append(valid_frame_detections)
        frames_vehicle_count.append(vehicle_count)
    
    print('Post-processing...')
    for i in tqdm(range(len(video_cap_images)-1)):
        # add contours to original frames
        frame = video_cap_images[i].copy()
        cv2.drawContours(frame, frames_valid_bboxes[i][0], -1, (127,200,0), 2)

        display_vehicle_count(frames_vehicle_count[i], frame)

        draw_detection_zone(DETECTION_ZONE, frame)

        for fd_center_point, fr_id in valid_frames_detections[i].items():
            draw_detection_box_id(str(fr_id), fd_center_point, frame)

        # Write the output frame to disk
        writer.write(frame)

        # https://stackoverflow.com/questions/35372700/whats-0xff-for-in-cv2-waitkey1
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
    writer.release()
    video_cap.release()
    print('Done')


if __name__ == '__main__':
    main(sys.argv[1:])



# python fd_approach.py --input_path ../videos/front_view.mp4 --output_path ../out_videos/automa --contour_area 2000 --dz_height_percentage 0.4
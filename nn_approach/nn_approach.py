import os
import sys
import time

import cv2
import numpy as np
from scipy import spatial
from tqdm import tqdm

# seed setting
np.random.seed(42)


weights_path = "yolo_config/yolov3.weights"
config_path = "yolo_config/yolov3.cfg"

try:
    net = cv2.dnn.readNetFromDarknet(
        config_path, 
        weights_path
    )
except cv2.error:
    print('YOLO config files are missing!')
    sys.exit()

INPUT_WIDTH, INPUT_HEIGHT = 416, 416

# Get the name of all layers of the network
ln = net.getLayerNames()
# Get the index of the output layers
ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]


COCO_LABELS = open("yolo_config/coco.names").read().strip().split("\n")

CHOSEN_LABELS = ["bicycle", "car", "motorbike", "bus", "truck", "train"]

# Initialize a list of colors to represent each possible class label
LABEL_COLORS = np.random.randint(
    0, 255, 
    size=(len(COCO_LABELS), 3), 
    dtype="uint8"
)


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


# PURPOSE: Draw all the detection boxes with a green dot at the center
# PARAMETERS: 
#   - box_ids - detection box ids
#   - boxes - all bounding boxes
#   - class_ids - detection class ids
#   - confidences - detection confidences
#   - frame - frame on which the detection boxes are displayed
def draw_detected_boxes(box_ids, boxes, class_ids, confidences, frame):
    if len(box_ids) > 0:
        for i in box_ids.flatten():
            # Extract the bounding box coordinates
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])

            # Draw a bounding box rectangle and label on the frame
            color = [int(c) for c in LABEL_COLORS[class_ids[i]]]

            cv2.rectangle(
                frame, 
                (x, y), 
                (x + w, y + h), 
                color, 
                2
            )

            text = "{}: {:.4f}".format(
                COCO_LABELS[class_ids[i]],
                confidences[i]
            )
            cv2.putText(
                frame, 
                text, 
                (x, y - 5), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.5, 
                color, 
                2
            )

            # Draw a green dot in the middle of the box
            cv2.circle(
                frame, 
                (x + (w // 2), y + (h // 2)), 
                2, 
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
#   - box_ids - detection box ids
#   - boxes - all bounding boxes
#   - class_ids - detection class ids
#   - vehicle_count - current vehicle count
#   - prev_frames_detections - all the vehicular detections of N previous frames (N = FRAMES_BEFORE_CURRENT)
#   - frames_defore_current - number of frames saved in history
#   - frame - frame on which the detection boxes are displayed
# RETURN: 
#   - vehicle_count - new vehicle count
#	- current_detections - current detections
def count_vehicles(box_ids, boxes, class_ids, vehicle_count, prev_frames_detections, detection_zone, frames_defore_current, frame):
    current_detections = {}

    if len(box_ids) > 0:
        for i in box_ids.flatten():
            # extract the bounding box coordinates
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])

            center_x = x + (w // 2)
            center_y = y + (h // 2)

            # When the detection is in CHOSEN_LABELS
            if COCO_LABELS[class_ids[i]] in CHOSEN_LABELS and \
               (center_x > detection_zone[0][0]) and \
               (center_x < detection_zone[1][0]) and \
               (center_y > detection_zone[0][1]) and \
               (center_y < detection_zone[1][1]):
                current_detections[(center_x, center_y)] = vehicle_count
                current_box = (center_x, center_y, w, h)

                # When the ID of the detection is not present in previous frames detections
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
                draw_detection_box_id(str(current_box_id), current_box_center, frame)

    return vehicle_count, current_detections


# PURPOSE: Initializing the video writer with the output video path and the same number
#   of fps, width and height as the source video
# PARAMETERS: 
#   - video_width - video width
#   - video_height - video height
#   - video_cap - video stream
#   - out_path - output video path
# RETURN: The initialized video writer
def initialize_video_writer(video_width, video_height, video_fps, out_path):
    video_writer = cv2.VideoWriter(
        out_path, 
        cv2.VideoWriter_fourcc(*'MJPG'), # XVID
        video_fps, 
        (video_width, video_height))

    return video_writer


def main(argv):
    if len(argv) < 4 or '--input_path' not in argv or '--output_path' not in argv:
        print(
            '\n================================================================ \n'
            'USAGE:\n'
            '  python nn_approach.py --input_path <path_to_video_file> --output_path <output_video_dir>\n'
            'OPTIONS: \n'
            '  Detection zone width percentage (calculated from the bottom of the image) \n'
            '    --dz_height_percentage <e.g. 0.8> \n'
            '  Detection zone width percentage (calculated form the left side of the image) \n'
            '    --dz_width_percentage <e.g. 0.4> \n'
            '  Threshold of confindence in detection \n'
            '    --confidence_threshold <e.g. 0.5> \n'
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
        confidence_threshold_argv = argv.index('--confidence_threshold')
        confidence_threshold = argv[confidence_threshold_argv+1]
        # if not confidence_threshold.isdigit():
        #     print('Invalid input!')
        #     sys.exit()
        DETECTION_CONFIDENCE = float(confidence_threshold)
    except ValueError:
        DETECTION_CONFIDENCE = 0.5
    except IndexError:
        print('Invalid input!')
        sys.exit()

    # print(video_in_path)
    # print(video_out_path)
    # print(DETECTION_ZONE_H_PERCENTAGE)
    # print(DETECTION_ZONE_W_PERCENTAGE)
    # print(THRESHOLD)

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

    # DETECTION_CONFIDENCE = 0.5
    DETECTION_THRESHOLD = 0.3
    FRAMES_BEFORE_CURRENT = round(video_fps / 4)    # (e.g. 60/4 = 15)
    DISPLAY_WHILE_PROCESSING = False

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

    # Main loop over video frames
    print('Processing...')
    for _ in tqdm(range(video_frames_count)):
        # Initialization for each iteration
        boxes, confidences, class_ids = [], [], []

        # Read the next video frame
        (grabbed, frame) = video_cap.read()

        # If the frame was not grabbed, then we have reached the end of the stream
        if not grabbed:
            break

        # Construct a blob from the input frame and then perform a forward
        # pass of the YOLO object detector, giving us our bounding boxes
        # and associated confidences
        blob = cv2.dnn.blobFromImage(
            frame, 
            1 / 255.0, 
            (INPUT_WIDTH, INPUT_HEIGHT), 
            swapRB=True, 
            crop=False
        )
        net.setInput(blob)
        layer_outputs = net.forward(ln)

        # Loop over each layer outputs
        for output_detections in layer_outputs:
            # Loop over each of detection
            for i, detection in enumerate(output_detections):
                # detection[0:4] return center_x, center_y, width, height
                #   detection[5:] returns the confidences score for each class
                detection_confidences = detection[5:]
                class_id = np.argmax(detection_confidences)
                confidence = detection_confidences[class_id]

                # Filter out weak predictions
                if confidence > DETECTION_CONFIDENCE:
                    # Scale the bounding box coordinates back relative to
                    # the size of the image, keeping in mind that YOLO actually 
                    # returns the center (x, y)-coordinates of the bounding box 
                    # followed by the boxes' width and height
                    box = detection[0:4] * np.array([video_width, video_height, video_width, video_height])
                    (center_x, center_y, width, height) = box.astype("int")

                    # Get the top and left corner of the bounding box
                    x = int(center_x - (width / 2))
                    y = int(center_y - (height / 2))

                    # Update: bounding box coordinates, confidences, and class ids lists
                    boxes.append([x, y, int(width), int(height)])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        # Apply non-maxima suppression to suppress weak, overlapping bounding boxes
        box_ids = cv2.dnn.NMSBoxes(
            bboxes=boxes, 
            scores=confidences, 
            score_threshold=DETECTION_CONFIDENCE, 
            nms_threshold=DETECTION_THRESHOLD
        )

        # Draw detection box
        draw_detected_boxes(
            box_ids=box_ids, 
            boxes=boxes, 
            class_ids=class_ids, 
            confidences=confidences, 
            frame=frame
        )

        vehicle_count, current_detections = count_vehicles(
            box_ids=box_ids, 
            boxes=boxes, 
            class_ids=class_ids, 
            vehicle_count=vehicle_count, 
            prev_frames_detections=prev_frames_detections,
            detection_zone=DETECTION_ZONE,
            frames_defore_current=FRAMES_BEFORE_CURRENT,
            frame=frame
        )
        
        draw_detection_zone(DETECTION_ZONE, frame)

        # Display Vehicle Count
        display_vehicle_count(vehicle_count, frame)

        # Write the output frame to disk
        writer.write(frame)

        if DISPLAY_WHILE_PROCESSING:
            cv2.imshow('Frame', frame)

        # https://stackoverflow.com/questions/35372700/whats-0xff-for-in-cv2-waitkey1
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        # Updating prev_frames_detections
        # Removing the first frame from the list
        prev_frames_detections.pop(0)
        prev_frames_detections.append(current_detections)
        
    writer.release()
    video_cap.release()
    print('Done')


if __name__ == '__main__':
    main(sys.argv[1:])


# python nn_approach.py --input_path ../videos/front_view.mp4 --output_path ../out_videos/automa --confidence_threshold 0.5 --dz_height_percentage 0.4 --dz_width_percentage 1.0
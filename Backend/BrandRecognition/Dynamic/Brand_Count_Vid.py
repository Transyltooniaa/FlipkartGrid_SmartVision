import cv2
import numpy as np
from collections import deque, defaultdict
from ultralytics import YOLO



def iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])

    iou = intersection / float(area1 + area2 - intersection)
    return iou

def smooth_box(box_history):
    if not box_history:
        return None
    return np.mean(box_history, axis=0)

def process_video(input_path, output_path):
    model = YOLO('Weights/kitkat_s.pt')
    cap = cv2.VideoCapture(input_path)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    detected_items = {}
    frame_count = 0

    detections_history = defaultdict(lambda: defaultdict(int))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        if frame_count % 5 == 0:
            results = model(frame)

            current_frame_detections = []

            for r in results:
                boxes = r.boxes
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    conf = box.conf.item()
                    cls = int(box.cls.item())
                    brand = model.names[cls]

                    current_frame_detections.append((brand, [x1, y1, x2, y2], conf))

            for brand, box, conf in current_frame_detections:
                matched = False
                for item_id, item_info in detected_items.items():
                    if iou(box, item_info['smoothed_box']) > 0.5:
                        item_info['frames_detected'] += 1
                        item_info['total_conf'] += conf
                        item_info['box_history'].append(box)
                        if len(item_info['box_history']) > 10:
                            item_info['box_history'].popleft()
                        item_info['smoothed_box'] = smooth_box(item_info['box_history'])
                        item_info['last_seen'] = frame_count
                        matched = True
                        break

                if not matched:
                    item_id = len(detected_items)
                    detected_items[item_id] = {
                        'brand': brand,
                        'box_history': deque([box], maxlen=10),
                        'smoothed_box': box,
                        'frames_detected': 1,
                        'total_conf': conf,
                        'last_seen': frame_count
                    }

                detections_history[brand][frame_count] += 1


        for item_id, item_info in list(detected_items.items()):
            if frame_count - item_info['last_seen'] > fps * 2:  # 2 seconds
                del detected_items[item_id]
                continue

            if item_info['smoothed_box'] is not None:
                alpha = 0.3
                current_box = item_info['smoothed_box']
                target_box = item_info['box_history'][-1] if item_info['box_history'] else current_box
                interpolated_box = [
                    current_box[i] * (1 - alpha) + target_box[i] * alpha
                    for i in range(4)
                ]
                item_info['smoothed_box'] = interpolated_box

                x1, y1, x2, y2 = map(int, interpolated_box)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"{item_info['brand']}",
                            (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        out.write(frame)

    cap.release()
    out.release()

    total_frames = frame_count
    confirmed_items = {}
    for brand, frame_counts in detections_history.items():
        detection_frames = len(frame_counts)
        if detection_frames > total_frames * 0.1:
            avg_count = sum(frame_counts.values()) / detection_frames
            confirmed_items[brand] = round(avg_count)

    return confirmed_items

def annotate_video(input_video):
    output_path = 'annotated_output.mp4'
    confirmed_items = process_video(input_video, output_path)

    item_list = [(brand, quantity) for brand, quantity in confirmed_items.items()]

    status_message = "Video processed successfully!"

    return output_path, item_list, status_message

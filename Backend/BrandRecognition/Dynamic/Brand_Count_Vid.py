import cv2
import numpy as np
from collections import deque, defaultdict
from ultralytics import YOLO

def comprehensive_frame_reduction(
    input_path, 
    output_path, 
    target_frame_count=100, 
    scene_change_weight=0.5, 
    motion_weight=0.3, 
    color_weight=0.2
):
    """
    Comprehensively reduce video frames while maintaining video content representation.
    
    Args:
        input_path (str): Path to input video
        output_path (str): Path to output video
        target_frame_count (int): Desired number of frames in reduced video
        scene_change_weight (float): Weight for scene change importance
        motion_weight (float): Weight for motion importance
        color_weight (float): Weight for color distribution importance
    
    Returns:
        List of selected frame indices
    """
    # Open the video
    cap = cv2.VideoCapture(input_path)
    
    # Video properties
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Output video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Preprocessing
    frames = []
    frame_grays = []
    frame_hists = []
    
    # Read all frames
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frames.append(frame)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame_grays.append(gray)
        
        # Compute color histogram
        hist = cv2.calcHist([frame], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
        frame_hists.append(hist.flatten())
    
    cap.release()
    
    # Compute frame importance scores
    importance_scores = np.zeros(len(frames))
    
    # Scene change detection
    scene_changes = np.zeros(len(frames))
    for i in range(1, len(frames)):
        # Compute frame difference
        frame_diff = cv2.absdiff(frame_grays[i-1], frame_grays[i])
        scene_changes[i] = np.sum(frame_diff) / (width * height)
    
    # Motion estimation (using frame differences)
    motion_scores = np.zeros(len(frames))
    for i in range(2, len(frames)):
        # Compute motion between consecutive frames
        motion_diff = cv2.absdiff(frame_grays[i-2], frame_grays[i])
        motion_scores[i] = np.sum(motion_diff) / (width * height)
    
    # Color distribution variance
    color_variance = np.zeros(len(frames))
    for i in range(len(frames)):
        # Compute color histogram difference from average
        color_variance[i] = np.linalg.norm(frame_hists[i] - np.mean(frame_hists, axis=0))
    
    # Normalize scores
    scene_changes = (scene_changes - scene_changes.min()) / (scene_changes.max() - scene_changes.min())
    motion_scores = (motion_scores - motion_scores.min()) / (motion_scores.max() - motion_scores.min())
    color_variance = (color_variance - color_variance.min()) / (color_variance.max() - color_variance.min())
    
    # Compute importance scores
    importance_scores = (
        scene_change_weight * scene_changes +
        motion_weight * motion_scores +
        color_weight * color_variance
    )
    
    # Ensure even distribution across video
    segment_size = len(frames) // target_frame_count
    selected_frames = []
    
    for segment in range(target_frame_count):
        start = segment * segment_size
        end = (segment + 1) * segment_size if segment < target_frame_count - 1 else len(frames)
        
        # Find the most important frame in this segment
        segment_scores = importance_scores[start:end]
        best_frame_idx = np.argmax(segment_scores) + start
        
        selected_frames.append(best_frame_idx)
        
        # Write selected frames to output video
        out.write(frames[best_frame_idx])
    
    out.release()
    
    print(f"Reduced from {total_frames} to {len(selected_frames)} frames")
    return selected_frames


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
    
    comprehensive_frame_reduction(
        input_video,
        'reduced_video.mp4',
        target_frame_count=100,  
        scene_change_weight=0.5,
        motion_weight=0.3,
        color_weight=0.2
    )

    confirmed_items = process_video('reduced_video.mp4', output_path)

    item_list = [(brand, quantity) for brand, quantity in confirmed_items.items()]

    status_message = "Video processed successfully!"

    return output_path, item_list, status_message

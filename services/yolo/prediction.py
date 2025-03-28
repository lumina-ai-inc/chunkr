import numpy as np
import logging
import collections
from sklearn.cluster import KMeans
from typing import List, Tuple

from models import BoundingBox, BoundingBoxOutput

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def map_yolo_to_segment_type(yolo_class: int, box_y: float, img_height: float, is_first_title: bool = True) -> int:
    # Map YOLO classes to SegmentType from Rust code
    if yolo_class == 0:  # title
        if is_first_title:
            return 10  # Title (only for the first title)
        else:
            return 3   # Section Header (for subsequent titles)
    elif yolo_class == 1:  # plain text
        return 9   # Text
    elif yolo_class == 2:  # abandon
        # Check position to determine if header or footer
        relative_position = box_y / img_height
        if relative_position < 0.2:
            return 5  # PageHeader
        elif relative_position > 0.8:
            return 4  # PageFooter
        else:
            return 9  # Text (default if not clearly header/footer)
    elif yolo_class == 3:  # figure
        return 6   # Picture
    elif yolo_class == 4:  # figure_caption
        return 0   # Caption
    elif yolo_class == 5:  # table
        return 8   # Table
    elif yolo_class == 6:  # table_caption
        return 0   # Caption
    elif yolo_class == 7:  # table_footnote
        return 1   # Footnote
    elif yolo_class == 8:  # isolate_formula
        return 2   # Formula
    elif yolo_class == 9:  # formula_caption
        return 0   # Caption
    else:
        return 9   # Default to Text

def merge_colliding_predictions(boxes: List[BoundingBox], scores: List[float], classes: List[int], score_threshold: float = 0.2, overlap_threshold: float = 0.1) -> tuple[List[BoundingBox], List[float], List[int]]:
    valid_indices = [i for i, score in enumerate(scores) if score >= score_threshold]
    if not valid_indices:
        return boxes, scores, classes
    
    filtered_boxes = [boxes[i] for i in valid_indices]
    filtered_scores = [scores[i] for i in valid_indices]
    filtered_classes = [classes[i] for i in valid_indices]
    
    merged_boxes, merged_scores, merged_classes = [], [], []
    
    while filtered_boxes:
        box1 = filtered_boxes.pop(0)
        score1 = filtered_scores.pop(0)
        class1 = filtered_classes.pop(0)
        
        to_merge_indices = []
        for i, box2 in enumerate(filtered_boxes):
            intersection = (
                max(0, min(box1.x2, box2.x2) - max(box1.x1, box2.x1)) *
                max(0, min(box1.y2, box2.y2) - max(box1.y1, box2.y1))
            )
            area1 = (box1.x2 - box1.x1) * (box1.y2 - box1.y1)
            area2 = (box2.x2 - box2.x1) * (box2.y2 - box2.y1)
            min_area = min(area1, area2)
            
            if min_area > 0 and (intersection / min_area) > overlap_threshold:
                to_merge_indices.append(i)
        
        if to_merge_indices:
            merge_boxes = [box1] + [filtered_boxes[i] for i in to_merge_indices]
            merge_scores = [score1] + [filtered_scores[i] for i in to_merge_indices]
            merge_classes = [class1] + [filtered_classes[i] for i in to_merge_indices]
            
            for i in reversed(to_merge_indices):
                filtered_boxes.pop(i)
                filtered_scores.pop(i)
                filtered_classes.pop(i)
            
            x1 = min(box.x1 for box in merge_boxes)
            y1 = min(box.y1 for box in merge_boxes)
            x2 = max(box.x2 for box in merge_boxes)
            y2 = max(box.y2 for box in merge_boxes)
            
            class_counts = collections.Counter(merge_classes)
            final_class = class_counts.most_common(1)[0][0]
            
            merged_boxes.append(BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2))
            merged_scores.append(max(merge_scores))
            merged_classes.append(final_class)
        else:
            merged_boxes.append(box1)
            merged_scores.append(score1)
            merged_classes.append(class1)
    
    # Merge small boxes into larger ones if overlap > 90%
    final_boxes, final_scores, final_classes = [], [], []
    for i, box1 in enumerate(merged_boxes):
        merged = False
        for j, box2 in enumerate(merged_boxes):
            if i != j:
                intersection = (
                    max(0, min(box1.x2, box2.x2) - max(box1.x1, box2.x1)) *
                    max(0, min(box1.y2, box2.y2) - max(box1.y1, box2.y1))
                )
                area1 = (box1.x2 - box1.x1) * (box1.y2 - box1.y1)
                if area1 > 0 and (intersection / area1) > 0.9:
                    merged = True
                    break
        if not merged:
            final_boxes.append(box1)
            final_scores.append(merged_scores[i])
            final_classes.append(merged_classes[i])
    
    return final_boxes, final_scores, final_classes

def apply_reading_order(boxes, scores, classes, image_size, score_threshold):
    if not boxes:
        return boxes, scores, classes, image_size

    def is_wide_element(box, page_width, threshold=0.7):
        # Check if element spans across multiple column boundaries
        width_ratio = box.width / page_width
        center_x = box.left + box.width / 2
        left_edge = box.left
        right_edge = box.left + box.width
        
        # If element takes up significant width OR crosses column centers, treat as wide
        return (width_ratio > threshold or 
                (left_edge < page_width * 0.4 and right_edge > page_width * 0.6))

    def get_column_assignment(box, col_boundaries):
        center_x = box.left + box.width / 2
        for i, (left, right) in enumerate(col_boundaries):
            if left <= center_x <= right:
                return i
        return 0

    # Convert BoundingBox objects to BoundingBoxOutput for processing
    bbox_outputs = []
    for box in boxes:
        bbox_outputs.append(BoundingBoxOutput(
            left=box.x1,
            top=box.y1,
            width=box.x2 - box.x1,
            height=box.y2 - box.y1
        ))

    # Calculate page width
    page_width = max(box.left + box.width for box in bbox_outputs) if bbox_outputs else 1000

    # Determine column boundaries using clustering
    centers_x = np.array([[box.left + box.width / 2] for box in bbox_outputs])
    
    # Use the elbow method to find the optimal number of clusters
    distortions = []
    K = range(1, min(10, len(centers_x) + 1))  # Ensure K is within valid range
    for k in K:
        if k == 1:
            distortions.append(0)
            continue
        kmeans = KMeans(n_clusters=k, n_init="auto")
        kmeans.fit(centers_x)
        distortions.append(kmeans.inertia_)

    # Find the elbow point
    optimal_k = 1
    if len(distortions) > 2:
        for i in range(1, len(distortions) - 1):
            if distortions[i] - distortions[i + 1] < distortions[i - 1] - distortions[i]:
                optimal_k = i + 1
                break

    # Perform clustering with the optimal number of clusters
    col_boundaries = []
    if optimal_k > 1:
        kmeans = KMeans(n_clusters=optimal_k, n_init="auto")
        kmeans.fit(centers_x)
        for i in range(optimal_k):
            cluster_points = centers_x[kmeans.labels_ == i]
            if cluster_points.size > 0:
                col_boundaries.append((cluster_points.min(), cluster_points.max()))

    # Ensure col_boundaries is not empty and sort them
    if not col_boundaries:
        col_boundaries = [(0, page_width)]
    col_boundaries.sort(key=lambda x: x[0])  # Sort by the left boundary

    # Process segments
    segments = []
    for i, (box, score, class_id) in enumerate(zip(bbox_outputs, scores, classes)):
        if score > score_threshold:
            is_wide = is_wide_element(box, page_width)
            segments.append({
                'idx': i,
                'box': box,
                'score': score,
                'class': class_id,
                'is_wide': is_wide,
                'center_x': box.left + box.width / 2,
                'center_y': box.top + box.height / 2
            })

    # Separate headers and footers
    headers = [s for s in segments if s['class'] == 5]
    footers = [s for s in segments if s['class'] in [1, 4]]
    body = [s for s in segments if s['class'] not in [1, 4, 5]]

    def process_body_segments(segments):
        segments.sort(key=lambda s: s['box'].top)
        columns = [[] for _ in range(optimal_k)]
        current_y = float('-inf')
        temp_segments = []
        result = []

        for segment in segments:
            if abs(segment['box'].top - current_y) > 20:
                if temp_segments:
                    if any(s['is_wide'] for s in temp_segments):
                        for col in columns:
                            result.extend(col)
                        columns = [[] for _ in range(optimal_k)]
                        result.extend(temp_segments)
                    else:
                        for s in temp_segments:
                            if s['is_wide']:
                                for col in columns:
                                    result.extend(col)
                                columns = [[] for _ in range(optimal_k)]
                                result.append(s)
                            else:
                                col = get_column_assignment(s['box'], col_boundaries)
                                columns[col].append(s)
                temp_segments = [segment]
                current_y = segment['box'].top
            else:
                temp_segments.append(segment)

        if temp_segments:
            if any(s['is_wide'] for s in temp_segments):
                for col in columns:
                    result.extend(col)
                result.extend(temp_segments)
            else:
                for s in temp_segments:
                    col = get_column_assignment(s['box'], col_boundaries)
                    columns[col].append(s)

        for col in columns:
            result.extend(col)
            
        return result

    ordered_segments = []
    ordered_segments.extend(sorted(headers, key=lambda s: s['box'].top))
    ordered_segments.extend(process_body_segments(body))
    ordered_segments.extend(sorted(footers, key=lambda s: s['box'].top))

    if ordered_segments:
        final_indices = [s['idx'] for s in ordered_segments]
        return [boxes[i] for i in final_indices], [scores[i] for i in final_indices], [classes[i] for i in final_indices], image_size
    
    return boxes, scores, classes, image_size

def get_reading_order_and_merge(boxes: List[BoundingBox], scores: List[float], classes: List[int], image_size: Tuple[int, int], score_threshold: float, overlap_threshold: float) -> Tuple[List[BoundingBox], List[float], List[int], Tuple[int, int]]:
    # First merge overlapping predictions
    merged_boxes, merged_scores, merged_classes = merge_colliding_predictions(boxes, scores, classes, score_threshold, overlap_threshold)
    
    # Then apply reading order
    ordered_boxes, ordered_scores, ordered_classes, image_size = apply_reading_order(
        merged_boxes, merged_scores, merged_classes, image_size, score_threshold
    )
    
    return ordered_boxes, ordered_scores, ordered_classes, image_size


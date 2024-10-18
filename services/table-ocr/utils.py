import torch
from typing import List
from torchvision.transforms import functional as F
from pydantic import BaseModel, Field


class BoundingBox(BaseModel):
    x1: float
    y1: float
    x2: float
    y2: float

    def to_dict(self):
        return {
            'left': self.x1,
            'top': self.y1,
            'width': self.x2 - self.x1,
            'height': self.y2 - self.y1
        }


class Cell(BaseModel):
    column: BoundingBox
    cell: BoundingBox
    col_span: int = Field(default=1)
    row_span: int = Field(default=1)

    def to_dict(self):
        return {
            'column': self.column.to_dict(),
            'cell': self.cell.to_dict(),
            'col_span': self.col_span,
            'row_span': self.row_span
        }

class Row(BaseModel):
    row: BoundingBox
    cells: List[Cell]
    cell_count: int
    confidence: float

    def to_dict(self):
        return {
            'row': self.row.to_dict(),
            'cells': [cell.to_dict() for cell in self.cells],
            'cell_count': self.cell_count,
            'confidence': self.confidence
        }


class TableStructure(BaseModel):
    rows: List[Row]

    def to_dict(self):
        return {
            'rows': [row.to_dict() for row in self.rows]
        }


class MaxResize:
    def __init__(self, max_size):
        self.max_size = max_size

    def __call__(self, image):
        width, height = image.size
        if width > height:
            if width > self.max_size:
                height = int(height * self.max_size / width)
                width = self.max_size
        else:
            if height > self.max_size:
                width = int(width * self.max_size / height)
                height = self.max_size
        return F.resize(image, (height, width))


def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h), (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=1)


def calculate_iou(box1, box2):
    """Calculate Intersection over Union (IoU) of two bounding boxes."""
    x1, y1, x2, y2 = box1
    x1_p, y1_p, x2_p, y2_p = box2

    inter_x1 = max(x1, x1_p)
    inter_y1 = max(y1, y1_p)
    inter_x2 = min(x2, x2_p)
    inter_y2 = min(y2, y2_p)

    inter_width = max(inter_x2 - inter_x1, 0)
    inter_height = max(inter_y2 - inter_y1, 0)
    inter_area = inter_width * inter_height

    area_box1 = (x2 - x1) * (y2 - y1)
    area_box2 = (x2_p - x1_p) * (y2_p - y1_p)

    union_area = area_box1 + area_box2 - inter_area
    if union_area == 0:
        return 0
    return inter_area / union_area


def rescale_bboxes(out_bbox, size):
    img_w, img_h = size
    b = box_cxcywh_to_xyxy(out_bbox)
    b = b * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32)
    return b


def outputs_to_objects(outputs, img_size, id2label):
    m = outputs.logits.softmax(-1).max(-1)
    pred_labels = list(m.indices.detach().cpu().numpy())[0]
    pred_scores = list(m.values.detach().cpu().numpy())[0]
    pred_bboxes = outputs['pred_boxes'].detach().cpu()[0]
    pred_bboxes = [elem.tolist()
                   for elem in rescale_bboxes(pred_bboxes, img_size)]

    objects = []
    for label, score, bbox in zip(pred_labels, pred_scores, pred_bboxes):
        class_label = id2label[int(label)]
        if not class_label == 'no object':
            objects.append({'label': class_label, 'score': float(score),
                            'bbox': [float(elem) for elem in bbox]})

    return objects


def get_cell_coordinates_by_row(table_data, merge_threshold=0.12, raw_output=False):
    rows = [entry for entry in table_data if entry['label'] == 'table row']
    columns = [entry for entry in table_data if entry['label'] == 'table column']
    spanning_cells = [entry for entry in table_data if entry['label'] in [
        'table spanning cell', 'table spanning row', 'table spanning column']]

    rows.sort(key=lambda x: x['bbox'][1])
    columns.sort(key=lambda x: x['bbox'][0])

    table_structure = []

    for row in rows:
        row_cells = []
        row_x1, row_y1, row_x2, row_y2 = row['bbox']

        for column in columns:
            col_x1, col_y1, col_x2, col_y2 = column['bbox']
            cell_bbox = [col_x1, row_y1, col_x2, row_y2]

            spanning_cell = find_best_spanning_cell(cell_bbox, spanning_cells)
            if spanning_cell:
                col_span, row_span = calculate_span(spanning_cell['bbox'], columns, rows)
                cell_info = Cell(
                    column=BoundingBox(x1=spanning_cell['bbox'][0], y1=spanning_cell['bbox'][1],
                                       x2=spanning_cell['bbox'][2], y2=spanning_cell['bbox'][3]),
                    cell=BoundingBox(x1=spanning_cell['bbox'][0], y1=spanning_cell['bbox'][1],
                                     x2=spanning_cell['bbox'][2], y2=spanning_cell['bbox'][3]),
                    col_span=col_span,
                    row_span=row_span
                )
            else:
                cell_info = Cell(
                    column=BoundingBox(x1=column['bbox'][0], y1=column['bbox'][1],
                                       x2=column['bbox'][2], y2=column['bbox'][3]),
                    cell=BoundingBox(x1=cell_bbox[0], y1=cell_bbox[1],
                                     x2=cell_bbox[2], y2=cell_bbox[3]),
                    col_span=1,
                    row_span=1
                )

            row_cells.append(cell_info)

        table_structure.append(Row(
            row=BoundingBox(x1=row['bbox'][0], y1=row['bbox'][1],
                            x2=row['bbox'][2], y2=row['bbox'][3]),
            cells=row_cells,
            cell_count=len(row_cells),
            confidence=row['score']
        ))

    table_structure.sort(key=lambda x: x.row.y1)

    return table_structure

def calculate_span(cell_bbox, columns, rows):
    col_span = sum(1 for col in columns if calculate_iou(cell_bbox, col['bbox']) > 0.5)
    row_span = sum(1 for row in rows if calculate_iou(cell_bbox, row['bbox']) > 0.5)
    return col_span, row_span

def find_best_spanning_cell(cell_bbox, spanning_cells):
    max_iou = 0
    best_spanning_cell = None
    for spanning_cell in spanning_cells:
        iou = calculate_iou(cell_bbox, spanning_cell['bbox'])
        if iou > max_iou:
            max_iou = iou
            best_spanning_cell = spanning_cell
    return best_spanning_cell if max_iou > 0.005 else None


def merge_boxes(boxes, iou_threshold=0.5):
    """
    Merge overlapping boxes based on IoU.

    Args:
        boxes (list): List of bounding boxes with 'label', 'score', and 'bbox'.
        iou_threshold (float): Threshold to decide whether boxes overlap.

    Returns:
        list: Merged bounding boxes.
    """
    if not boxes:
        return []

    # Sort the boxes by the y-coordinate (top of the box)
    boxes = sorted(boxes, key=lambda x: x['bbox'][1])
    merged = []

    while boxes:
        current = boxes.pop(0)
        to_merge = [current]

        boxes_to_remove = []
        for box in boxes:
            iou = calculate_iou(current['bbox'], box['bbox'])
            if iou > iou_threshold:
                to_merge.append(box)
                boxes_to_remove.append(box)

        for box in boxes_to_remove:
            boxes.remove(box)

        avg_bbox = [
            sum(box['bbox'][i] for box in to_merge) / len(to_merge)
            for i in range(4)
        ]
        merged.append({
            'label': current['label'],
            'score': sum(box['score'] for box in to_merge) / len(to_merge),
            'bbox': avg_bbox
        })

    return merged

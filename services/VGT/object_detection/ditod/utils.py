import json
import os
import sys

import cv2
import numpy as np
from shapely.geometry import Polygon
from tabulate import tabulate


def get_image_path(image_dir, file_name_wo_ext):
    ext_list = ["", '.jpg', '.JPG', '.png', '.PNG', ".jpeg"]
    image_path = None
    for ext in ext_list:
        image_path_tmp = os.path.join(image_dir, file_name_wo_ext + ext)
        if os.path.exists(image_path_tmp):
            image_path = image_path_tmp
            break
    return image_path


def visual_badcase(image_path, pred_list, label_list, output_dir="visual_badcase", info=None, prefix=''):
    """
    """
    img = cv2.imread(image_path) if os.path.exists(image_path) is not None else None
    if img is None:
        print("--> Warning: skip, given iamge NOT exists: {}".format(image_path))
        return None

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    font = cv2.FONT_HERSHEY_SIMPLEX
    for label in label_list:
        points, class_id = label["poly"], label["category_id"]
        pts = np.array(points).reshape((1, -1, 2)).astype(np.int32)
        cv2.polylines(img, pts, isClosed=True, color=(0, 255, 0), thickness=3)
        cv2.putText(img, "gt:" + str(class_id), tuple(pts[0][0].tolist()), font, 1, (0, 255, 0), 2)

    for label in pred_list:
        points, class_id = label["poly"], label["category_id"]
        pts = np.array(points).reshape((1, -1, 2)).astype(np.int32)
        cv2.polylines(img, pts, isClosed=True, color=(255, 0, 0), thickness=3)
        cv2.putText(img, "pred:" + str(class_id), tuple(pts[0][-1].tolist()), font, 1, (255, 0, 0), 2)

    if info is not None:
        cv2.putText(img, str(info), (40, 40), font, 1, (0, 0, 255), 2)
    output_path = os.path.join(output_dir, prefix + os.path.basename(image_path) + "_vis.jpg")
    cv2.imwrite(output_path, img)
    return output_path

def pub_load_gt_from_json(json_path):
    """
    """
    with open(json_path) as f:
        gt_info = json.load(f)
    gt_image_list = gt_info["images"]
    gt_anno_list = gt_info["annotations"]

    id_to_image_info = {}
    for image_item in gt_image_list:
        id_to_image_info[image_item['id']] = {
            "file_name": image_item['file_name'],
            "group_name": image_item.get("group_name", "huntie")
        }

    group_info = {}
    for annotation_item in gt_anno_list:
        image_info = id_to_image_info[annotation_item['image_id']]
        image_name, group_name = image_info["file_name"], image_info["group_name"]
        
        # import ipdb;ipdb.set_trace()
        if image_name == '15_103.tar_1705.05489.gz_main_12_ori.jpg':
            print(image_info["file_name"], annotation_item['image_id'])
            # import ipdb;ipdb.set_trace()

        if group_name not in group_info:
            group_info[group_name] = {}
        if image_name not in group_info[group_name]:
            group_info[group_name][image_name] = []
            
        box_xywh = annotation_item["bbox"]
        box_xyxy = [ box_xywh[0], box_xywh[1], box_xywh[0] + box_xywh[2], box_xywh[1] + box_xywh[3]]
        pts = np.round([ box_xyxy[0], box_xyxy[1], box_xyxy[2], box_xyxy[1], box_xyxy[2], box_xyxy[3], box_xyxy[0], box_xyxy[3] ])
        anno_info = {
            "category_id": annotation_item["category_id"],
            "poly": pts,
            "secondary_id": annotation_item.get("secondary_id", -1),
            "direction_id": annotation_item.get("direction_id", -1)
        }
        group_info[group_name][image_name].append(anno_info)

    group_info_str = ", ".join(["{}[{}]".format(k, len(v)) for k, v in group_info.items()])
    print("--> load {} groups: {}".format(len(group_info.keys()), group_info_str))
    return group_info

def load_gt_from_json(json_path):
    """
    """
    with open(json_path) as f:
        gt_info = json.load(f)
    gt_image_list = gt_info["images"]
    gt_anno_list = gt_info["annotations"]

    id_to_image_info = {}
    for image_item in gt_image_list:
        id_to_image_info[image_item['id']] = {
            "file_name": image_item['file_name'],
            "group_name": image_item.get("group_name", "huntie")
        }

    group_info = {}
    for annotation_item in gt_anno_list:
        image_info = id_to_image_info[annotation_item['image_id']]
        image_name, group_name = image_info["file_name"], image_info["group_name"]

        if group_name not in group_info:
            group_info[group_name] = {}
        if image_name not in group_info[group_name]:
            group_info[group_name][image_name] = []
        anno_info = {
            "category_id": annotation_item["category_id"],
            "poly": annotation_item["poly"],
            "secondary_id": annotation_item.get("secondary_id", -1),
            "direction_id": annotation_item.get("direction_id", -1)
        }
        group_info[group_name][image_name].append(anno_info)

    group_info_str = ", ".join(["{}[{}]".format(k, len(v)) for k, v in group_info.items()])
    print("--> load {} groups: {}".format(len(group_info.keys()), group_info_str))
    return group_info


def calc_iou(label, detect):
    label_box = []
    detect_box = []

    d_area = []
    for i in range(0, len(detect)):
        pred_poly = detect[i]["poly"]
        box_det = []
        for k in range(0, 4):
            box_det.append([pred_poly[2 * k], pred_poly[2 * k + 1]])
        detect_box.append(box_det)
        try:
            poly = Polygon(box_det)
            d_area.append(poly.area)
        except:
            print('invalid detects', pred_poly)
            exit(-1)

    l_area = []
    for i in range(0, len(label)):
        gt_poly = label[i]["poly"]
        box_gt = []
        for k in range(4):
            box_gt.append([gt_poly[2 * k], gt_poly[2 * k + 1]])
        label_box.append(box_gt)
        try:
            poly = Polygon(box_gt)
            l_area.append(poly.area)
        except:
            print('invalid detects', gt_poly)
            exit(-1)

    ol_areas = []
    for i in range(0, len(detect_box)):
        ol_areas.append([])
        poly1 = Polygon(detect_box[i])
        for j in range(0, len(label_box)):
            poly2 = Polygon(label_box[j])
            try:
                ol_area = poly2.intersection(poly1).area
            except:
                print('invaild pair', detect_box[i], label_box[j])
                ol_areas[i].append(0.0)
            else:
                ol_areas[i].append(ol_area)

    d_ious = [0.0] * len(detect_box)
    l_ious = [0.0] * len(label_box)
    for i in range(0, len(detect_box)):
        for j in range(0, len(label_box)):
            if int(label[j]["category_id"]) == int(detect[i]["category_id"]):
                iou = min(ol_areas[i][j] / (d_area[i] + 1e-10), ol_areas[i][j] / (l_area[j] + 1e-10))
            else:
                iou = 0
            d_ious[i] = max(d_ious[i], iou)
            l_ious[j] = max(l_ious[j], iou)
    return l_ious, d_ious


def eval(instance_info):
    img_name, label_info = instance_info
    label = label_info['gt']
    detect = label_info['det']
    l_ious, d_ious = calc_iou(label, detect)
    return [img_name, d_ious, l_ious, detect, label]


def static_with_class(rets, iou_thresh=0.7, is_verbose=True, map_info=None, src_image_dir=None, visualization_dir=None):
    if is_verbose:
        table_head = ['Class_id', 'Class_name', 'Pre_hit', 'Pre_num', 'GT_hit', 'GT_num', 'Precision', 'Recall', 'F-score']
    else:
        table_head = ['Class_id', 'Class_name', 'Precision', 'Recall', 'F-score']
    table_body = []
    class_dict = {}

    for i in range(len(rets)):
        img_name, d_ious, l_ious, detects, labels = rets[i]
        item_lv, item_dv, item_dm, item_lm = 0, 0, 0, 0
        for label in labels:
            item_lv += 1
            category_id = label["category_id"]
            if category_id not in class_dict:
                class_dict[category_id] = {}
                class_dict[category_id]['dm'] = 0
                class_dict[category_id]['dv'] = 0
                class_dict[category_id]['lm'] = 0
                class_dict[category_id]['lv'] = 0
            class_dict[category_id]['lv'] += 1

        for det in detects:
            item_dv += 1
            category_id = det["category_id"]
            if category_id not in class_dict:
                print("--> category_id not exists in gt: {}".format(category_id))
                continue
            class_dict[category_id]['dv'] += 1

        for idx, iou in enumerate(d_ious):
            if iou >= iou_thresh:
                item_dm += 1
                class_dict[detects[idx]["category_id"]]['dm'] += 1
        for idx, iou in enumerate(l_ious):
            if iou >= iou_thresh:
                item_lm += 1
                class_dict[labels[idx]["category_id"]]['lm'] += 1
        item_p = item_dm / (item_dv + 1e-6)
        item_r = item_lm / (item_lv + 1e-6)
        item_f = 2 * item_p * item_r / (item_p + item_r + 1e-6)

        
        if item_f < 0.97 and src_image_dir is not None:
            image_path = get_image_path(src_image_dir, os.path.basename(img_name))
            visualization_output = visualization_dir if visualization_dir is not None else "./visualization_badcase"
            item_info = "IOU{}, {}, {}, {}".format(iou_thresh, item_r, item_p, item_f)
            vis_path = visual_badcase(image_path, detects, labels, output_dir=visualization_output, info=item_info, prefix="{:02d}_".format(int(item_f * 100)))
            if is_verbose:
                print("--> info: save visualization at: {}".format(vis_path))

    dm, dv, lm, lv = 0, 0, 0, 0
    map_info = {} if map_info is None else map_info
    for key in class_dict.keys():
        dm += class_dict[key]['dm']
        dv += class_dict[key]['dv']
        lm += class_dict[key]['lm']
        lv += class_dict[key]['lv']
        p = class_dict[key]['dm'] / (class_dict[key]['dv'] + 1e-6)
        r = class_dict[key]['lm'] / (class_dict[key]['lv'] + 1e-6)
        fscore = 2 * p * r / (p + r + 1e-6)
        if is_verbose:
            table_body.append((key, map_info.get("primary_map", {}).get(str(key), str(key)), class_dict[key]['dm'],
                               class_dict[key]['dv'], class_dict[key]['lm'], class_dict[key]['lv'], p, r, fscore))
        else:
            table_body.append((key,  map_info.get(str(key), str(key)), p, r, fscore))

    p = dm / (dv + 1e-6)
    r = lm / (lv + 1e-6)
    f = 2 * p * r / (p + r + 1e-6)

    table_body_sorted = sorted(table_body, key=lambda x: int((x[0])))
    if is_verbose:
        table_body_sorted.append(('IOU_{}'.format(iou_thresh), 'average', dm, dv, lm, lv, p, r, f))
    else:
        table_body_sorted.append(('IOU_{}'.format(iou_thresh), 'average', p, r, f))
    print(tabulate(table_body_sorted, headers=table_head, tablefmt='pipe'))
    return [table_head] + table_body_sorted


def multiproc(func, task_list, proc_num=30, retv=True, progress_bar=False):
    from multiprocessing import Pool
    pool = Pool(proc_num)

    rets = []
    if progress_bar:
        import tqdm
        with tqdm.tqdm(total=len(task_list)) as t:
            for ret in pool.imap(func, task_list):
                rets.append(ret)
                t.update(1)
    else:
        for ret in pool.imap(func, task_list):
            rets.append(ret)

    pool.close()
    pool.join()

    if retv:
        return rets


def eval_and_show(label_dict, detect_dict, output_dir, iou_thresh=0.7, map_info=None,
                  src_image_dir=None, visualization_dir=None):
    """
    """
    evaluation_group_info = {}
    for group_name, gt_info in label_dict.items():
        group_pair_list = []
        for file_name, value_list in gt_info.items():
            if file_name not in detect_dict:
                print("--> missing pred:", file_name)
                continue
            group_pair_list.append([file_name, {'gt': gt_info[file_name], 'det': detect_dict[file_name]}])
        evaluation_group_info[group_name] = group_pair_list

    res_info_all = {}
    for group_name, group_pair_list in evaluation_group_info.items():
        print(" ------- group name: {} -----------".format(group_name))
        rets = multiproc(eval, group_pair_list, proc_num=16)
        group_name_map_info = map_info.get(group_name, None) if map_info is not None else None
        res_info = static_with_class(rets, iou_thresh=iou_thresh, map_info=group_name_map_info,
                                     src_image_dir=src_image_dir, visualization_dir=visualization_dir)
        res_info_all[group_name] = res_info

    evaluation_res_info_path = os.path.join(output_dir, "results_val.json")
    with open(evaluation_res_info_path, "w") as f:
        json.dump(res_info_all, f, ensure_ascii=False, indent=4)
    print("--> info: evaluation result is saved at {}".format(evaluation_res_info_path))


if __name__ == "__main__":

    if len(sys.argv) != 5:
        print("Usage: python {} gt_json_path pred_json_path output_dir iou_thresh".format(__file__))
        exit(-1)
    else:
        print('--> info: {}'.format(sys.argv))
        gt_json_path, pred_json_path, output_dir, iou_thresh = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]

    label_dict = load_gt_from_json(gt_json_path)
    with open(pred_json_path, "r") as f:
        detect_dict = json.load(f)

    src_image_dir = None
    eval_and_show(label_dict, detect_dict, output_dir, iou_thresh=iou_thresh, map_info=None,
                  src_image_dir=src_image_dir, visualization_dir=None)

import torch
import pandas as pd
from collections import Counter


def intersection_over_union(boxes_pred, boxes_label, box_format="midpoint"):

    if box_format == "midpoint":
        box1_x1 = boxes_pred[..., 0:1] - boxes_pred[..., 2:3]/2
        box1_y1 = boxes_pred[..., 1:2] - boxes_pred[..., 3:4]/2
        box1_x2 = boxes_pred[..., 0:1] + boxes_pred[..., 2:3]/2
        box1_y2 = boxes_pred[..., 1:2] + boxes_pred[..., 3:4]/2
        box2_x1 = boxes_label[..., 0:1] - boxes_label[..., 2:3]/2
        box2_y1 = boxes_label[..., 1:2] - boxes_label[..., 3:4]/2
        box2_x2 = boxes_label[..., 0:1] + boxes_label[..., 2:3]/2
        box2_y2 = boxes_label[..., 1:2] + boxes_label[..., 3:4]/2
        

    elif box_format == "corners":
        box1_x1 = boxes_pred[..., 0:1]
        box1_y1 = boxes_pred[..., 1:2]
        box1_x2 = boxes_pred[..., 2:3]
        box1_y2 = boxes_pred[..., 3:4]
        box2_x1 = boxes_label[..., 0:1]
        box2_y1 = boxes_label[..., 1:2]
        box2_x2 = boxes_label[..., 2:3]
        box2_y2 = boxes_label[..., 3:4]

    x1 = torch.min(box1_x1, box2_x1)
    y1 = torch.min(box1_y1, box2_y1)
    x2 = torch.min(box1_x2, box2_x2)
    y2 = torch.min(box1_y2, box2_y2)

    intersection = (x2 - x1).clamp(0)*(y2 - y1).clamp(0)

    box1_area = abs((box1_x2 - box1_x1)*(box1_y1 - box1_y2))
    box2_area = abs((box2_x2 - box2_x1)*(box2_y1 - box2_y2))

    return intersection / (box1_area + box2_area - intersection + 1e-6)


def nms(bboxes, iou_treshold, prob_threshold, box_format = "corners"):
    # predictions = [[class, prob, x1, y1, x2, y2], [...], [...]]
    bboxes = [box for box in bboxes if box[1]>prob_threshold]
    bboxes = sorted(bboxes, key=lambda x: x[1], reverse=True)
    bboxes_after_nms = []

    while bboxes:
        chosen_box = bboxes.pop(0)

        bboxes = [box for box in bboxes 
                  if box[0]!=chosen_box[0] 
                  or 
                  intersection_over_union(torch.tensor(chosen_box[2:]), torch.tensor(box[2:]), box_format=box_format) < iou_treshold
                  ]
        bboxes_after_nms.append(chosen_box)

    return bboxes_after_nms

def map(pred_boxes, true_boxes, iou_treshold=0.5, box_format="midpoint", num_classes=20):
    average_precisions = []
    epsilon = 1e-6
    for c in range(num_classes):
        detections = [box for box in pred_boxes if box[1]==c]
        ground_truths = [box for box in true_boxes if box[1]==c]

        amount_bboxes = Counter([gt[0] for gt in ground_truths])

        for key, val in amount_bboxes.items():
            amount_bboxes[key] = torch.zeros(val)
        
        detections.sort(key=lambda x: x[2], reverse=True)
        TP = torch.zeros((len(detections))) 
        FP = torch.zeros((len(detections)))
        total_true_boxes = len(ground_truths)

        if total_true_boxes==0:
            continue

        for detection_idx, detection in enumerate(detections):
            ground_truth_img = [
                bbox for bbox in ground_truths if bbox[0]==detection[0]
            ]
            num_gts = len(ground_truth_img)
            best_iou = 0

            for idx, gt in enumerate(ground_truth_img):
                iou = intersection_over_union(
                    torch.tensor(detection[3:]),
                    torch.tensor(gt[3:]),
                    box_format=box_format
                )

                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = idx

            if best_iou>iou_treshold:
                if amount_bboxes[detection[0]][best_gt_idx] == 0:
                    TP[detection]=1
                    amount_bboxes[detection[0]][best_gt_idx]=1
                else:
                    FP[detection_idx]=1
            else:
                FP[detection_idx]=1
        TP_cumsum = torch.cumsum(TP, dim=0)
        FP_cumsum = torch.cumsum(FP, dim=0)
        recalls = TP_cumsum/(total_true_boxes+epsilon)
        precisions = torch.divide(TP_cumsum, (TP_cumsum+FP_cumsum+epsilon))
        precisions = torch.cat((torch.tensor([1]), precisions))
        recalls = torch.cat((torch.tensor([0]), recalls))

        average_precisions.append(torch.trapz(precisions, recalls)) # trapz is for calculating the area so the roc curve
    return sum(average_precisions)/len(average_precisions)





  

    
import torch
import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset


class VOCDataset(Dataset):
    def __init__(
            self, csv_file, img_dir, label_dir, S=7, B=2, C=20, transform=None
    ):
        self.annotations = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.transform = transform
        self.S = S
        self.B = B
        self.C = C

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        label_path = os.path.join(
            self.label_dir, self.annotations.iloc[index, 1])
        boxes = []
        with open(label_path) as f:
            for label in f.readlines():
                class_label, x, y, width, height = [
                    float(x) if float(x) != int(float(x)) else int(x)
                    for x in label.replace("\n", "").split()

                ]
            boxes.append([class_label, x, y, width, height])

        img_path = os.path.join(self.img_dir, self.annotations.iloc[index, 0])
        image = Image.open(img_path)
        boxes = torch.tensor(boxes)

        if self.transform:
            image, boxes = self.transform(image, boxes)

        label_matrix = torch.zeros((self.S, self.S, self.C + 5*self.B))
        for box in boxes:
            class_label, x, y, width, height = box.tolist()
            class_label - int(class_label)
            i, j = int(self.S*y), int(self.S*x)
            x_cell, y_cell = self.S*x - j, self.S*y - i
            width_cell, height_cell = (
                width*self.S,
                height*self.S
            )

            if label_matrix[i, j, 20] == 0:
                label_matrix[i, j, 20] = 1
                box_coordinates = torch.tensor(
                    [x_cell, y_cell, width_cell, height_cell]
                )
                label_matrix[i, j, 21:25] = box_coordinates
                label_matrix[i, j, class_label] = 1

        return image, label_matrix


def intersection_over_union(boxes_pred, boxes_label, box_format="midpoint"):

    if box_format == "midpoint":

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

    intersection = (x2 - x1)*(y2 - y1).clamp(0)

    box1_area = abs((box1_x2 - box1_x1)*(box1_y1 - box1_y2))
    box2_area = abs((box2_x2 - box2_x1)*(box2_y1 - box2_y2))

    return intersection / (box1_area + box2_area - intersection + 1e-6)

#!/usr/bin/python3
import cv2
import numpy as np
import unittest


def iou_mean(y_true, y_pred):
    n_class = y_true.shape[-1]
    Iou = []
    for class_ in range(n_class):
        i = np.sum(y_pred[:, :, :, class_]*y_true[:, :, :, class_])
        u = np.sum(y_pred[:, :, :, class_]+y_true[:, :, :, class_] -
                   (y_pred[:, :, :, class_]*y_true[:, :, :, class_]))
        iou = i/u
        Iou.append(iou)
    return 1-np.mean(Iou)


class IOU_mean(unittest.TestCase):
    def test_iou_loss(self):
        true_lbl = cv2.imread("images/14.png", cv2.IMREAD_COLOR)
        true_lbl2 = cv2.imread("images/15.png", cv2.IMREAD_COLOR)
        true_lbl[np.where((true_lbl == [0, 0, 0]).all(axis=2))] = [255, 0, 0]
        true_lbl2[np.where((true_lbl2 == [0, 0, 0]).all(axis=2))] = [255, 0, 0]
        true_lbl = true_lbl / 255
        true_lbl2 = true_lbl2 / 255
        true_lbl = np.expand_dims(true_lbl, axis=0)
        true_lbl2 = np.expand_dims(true_lbl2, axis=0)

        print(iou_mean(true_lbl, true_lbl))
        print(iou_mean(true_lbl, true_lbl2))
        self.assertEqual(iou_mean(true_lbl, true_lbl), 0)
        self.assertGreater(iou_mean(true_lbl, true_lbl2), 0)
        self.assertLess(iou_mean(true_lbl, true_lbl),
                        iou_mean(true_lbl, true_lbl2))


if __name__ == "__main__":
    unittest.main()

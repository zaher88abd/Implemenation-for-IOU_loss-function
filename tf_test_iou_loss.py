import keras.backend as K
import tensorflow as tf
import numpy as np
import cv2


def iou_mean(y_true, y_pred):
    n_class = K.int_shape(y_true)[-1]
    Iou = []
    for class_ in range(n_class):
        i = tf.reduce_sum(tf.multiply(
            y_pred[:, :, :, class_], y_true[:, :, :, class_]))
        u = tf.reduce_sum(y_pred[:, :, :, class_] + y_true[:, :, :, class_] -
                          tf.multiply(y_pred[:, :, :, class_], y_true[:, :, :, class_]))
        Iou.append(tf.divide(i, u))
    Iou = tf.stack(Iou)
    return 1-K.mean(Iou)


class tf_iou_mean(tf.test.TestCase):
    def test_iou_loss(self):
        with self.test_session():
            true_lbl = cv2.imread("images/14.png", cv2.IMREAD_COLOR)
            true_lbl2 = cv2.imread("images/15.png", cv2.IMREAD_COLOR)
            true_lbl[np.where((true_lbl == [0, 0, 0]).all(axis=2))] = [
                255, 0, 0]
            true_lbl2[np.where((true_lbl2 == [0, 0, 0]).all(axis=2))] = [
                255, 0, 0]
            true_lbl = true_lbl / 255
            true_lbl2 = true_lbl2 / 255
            true_lbl = np.expand_dims(true_lbl, axis=0)
            true_lbl2 = np.expand_dims(true_lbl2, axis=0)
            true_lbl = tf.convert_to_tensor(true_lbl, tf.int16)
            true_lbl2 = tf.convert_to_tensor(true_lbl2, tf.int16)

            self.assertEqual(iou_mean(true_lbl, true_lbl).eval(), 0)
            self.assertGreater(iou_mean(true_lbl, true_lbl2).eval(), 0)
            self.assertLess(iou_mean(true_lbl, true_lbl).eval(),
                            iou_mean(true_lbl, true_lbl2).eval())


if __name__ == "__main__":
    tf.test.main()

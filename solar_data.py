# -*- encoding: utf-8 -*-

from __future__ import print_function
from keras.preprocessing.image import ImageDataGenerator, array_to_img, load_img, img_to_array
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import skimage.io as io
import skimage.transform as trans
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import cv2


# Shuffle two arrays synchronously
def shuffle_two_array(a, b):
    state = np.random.get_state()
    np.random.shuffle(a)
    np.random.set_state(state)
    np.random.shuffle(b)
    return a, b


def saltpepper_noise(image, proportion):  # Salt-and-pepper noise
    image_copy = image.copy()
    img_Y, img_X = image.shape
    X = np.random.randint(img_X, size=(int(proportion * img_X * img_Y),))
    Y = np.random.randint(img_Y, size=(int(proportion * img_X * img_Y),))
    image_copy[Y, X] = np.random.choice([0, 255], size=(int(proportion * img_X * img_Y),))

    sp_noise_plate = np.ones_like(image_copy) * 127

    sp_noise_plate[Y, X] = image_copy[Y, X]
    return image_copy


# load the batch image
def load_batch_image(img_path, target_size=(512, 512)):
    img = load_img(img_path, target_size)
    img = img_to_array(img)[:, :, 0]  # convertd image to numpy array, only read r channel
    img = img / 255.0
    img = np.expand_dims(img, axis=-1)
    return img


# load the whole data
def load_ha_data(train_dir, labels_dir, train_set, training_rate):
    val_size = 1 - training_rate
    test_size = val_size / 2
    all_imgaes_files = sorted(glob.glob(os.path.join(train_dir, '*')))  # Get a list of all iamges
    label_all_values = np.loadtxt(labels_dir)  # center coordinates and radius of solar disk
    label_all_adjust_values = []
    for iv in range(len(label_all_values)):
        label_all_adjust_values.append(label_all_values[iv, :])

    max_files_num = len(all_imgaes_files)  # Maximum number of images

    train_images = []
    train_labels = []
    valid_images = []
    valid_labels = []
    test_images = []
    test_labels = []
    fen = 1
    tag_8 = 1
    # TODO ready for the uniformming distribution
    while fen <= max_files_num:  # 8 : 1 : 1 train,val,test
        if tag_8 % 8 != 0:
            train_images.append(all_imgaes_files[fen - 1])
            train_labels.append(label_all_adjust_values[fen - 1])
            tag_8 = tag_8 + 1
            fen = fen + 1
        elif tag_8 % 8 == 0:
            train_images.append(all_imgaes_files[fen - 1])
            train_labels.append(label_all_adjust_values[fen - 1])

            valid_images.append(all_imgaes_files[fen])
            valid_labels.append(label_all_adjust_values[fen])

            test_images.append(all_imgaes_files[fen + 1])
            test_labels.append(label_all_adjust_values[fen + 1])
            tag_8 = 1
            fen = fen + 3

    train_images = np.array(train_images)
    train_labels = np.array(train_labels)
    valid_images = np.array(valid_images)
    valid_labels = np.array(valid_labels)
    test_images = np.array(test_images)
    test_labels = np.array(test_labels)

    if train_set == 'train':
        data = (train_images, train_labels)  # 训练集
    elif train_set == 'val':
        data = (valid_images, valid_labels)  # 验证集
    elif train_set == 'test':
        data = (test_images, test_labels)  # 验证集
    return data


def adjust_label_data(label):
    range_512 = 512
    label_1 = np.zeros(3 * range_512, int)
    label_1[int(np.round(label[0])) - 1] = 1
    label_1[int(np.round(label[1]) + range_512) - 1] = 1
    label_1[int(np.round(label[2]) + 2 * range_512) - 1] = 1
    return (label_1)


def adjust_data_lable(data):
    range_512 = 512
    data_1 = int(np.argmax(data[0:range_512]))
    data_2 = int(np.argmax(data[range_512:2 * range_512]))
    data_3 = int(np.argmax(data[2 * range_512:3 * range_512]))
    return [data_1, data_2, data_3]


# During training, the original image is visualized according to the label
def draw_circle(img, label, X_batches1, batch_size):
    X = img
    Y = label
    save_circle_path = "./data/train_aug_label/"
    save_aug_path = "./data/train_aug/"
    for batch_i in range(batch_size):
        img_X = X[batch_i, :, :, 0]
        label_Y = Y[batch_i, :]

        mm_data = img_X * 255.0
        cv2.imwrite(save_aug_path + str.split(X_batches1[batch_i], "\\")[1], mm_data)
        cv2.circle(mm_data, (int(label_Y[0]), int(label_Y[1])), int(label_Y[2]), (255, 255, 255), 2)
        cv2.imwrite(save_circle_path + str.split(X_batches1[batch_i], "\\")[1], mm_data)


#  Image augmentation; random sun disk location
def random_solardisk(X, Y):
    """
    :param X:
    :param Y:
    :return:
    """
    '''Four possibilities for new center location
    o:original center
    n:new center
————————————————————————————————————————————————————————————————————————————————————————————————
         ——————————————————————                |              ——————————————————————
         |                    |                |              |                    |
         |                    |                |              |                    |
         |    ————————————————|————————        |      ————————|—————————————       |
         |    |               |       |        |      |       |            |       |
         |    |         o*    |       |        |      |       |   o*       |       |
         ——————————————————————       |        |      |       |            |       |
              |                 n*    |        |      |    n* ——————————————————————
              |                       |        |      |                    |
              —————————————————————————        |      ——————————————————————
                        1                      |                2
————————————————————————————————————————————————————————————————————————————————————————————————
         ——————————————————————                |              ——————————————————————
         |                    |                |              |                    |
         |        n*          |                |              |            n*      |
         |    ————————————————|————————        |      ————————|—————————————       |
         |    |               |       |        |      |       |       o*   |       |
         |    |         o*    |       |        |      |       |            |       |
         ——————————————————————       |        |      |       |            |       |
              |                       |        |      |       ——————————————————————
              |                       |        |      |                    |
              —————————————————————————        |      ——————————————————————
                        4                      |                3
————————————————————————————————————————————————————————————————————————————————————————————————
    '''
    range_512 = 512
    range_10 = 10
    new_X = []
    new_Y = []
    for ic in range(len(Y[:, 0])):  # batch里面循环遍历
        try:
            original_image = X[ic, :, :, 0]
            original_x = int(np.round(Y[ic, 0]))
            original_y = int(np.round(Y[ic, 1]))
            solar_radius = int(round(Y[ic, 2]))

            # Random XY coordinates
            random_x = np.round(np.random.randint(solar_radius + range_10, range_512 - solar_radius - range_10))
            random_y = np.round(np.random.randint(solar_radius + range_10, range_512 - solar_radius - range_10))
            # The horizontal and vertical distance between the new coordinate and the original coordinate
            x_gap = np.round(random_x - original_x)
            y_gap = np.round(random_y - original_y)

            if x_gap >= 0 and y_gap >= 0:  # 1
                x_increase = range_512 - random_x
                y_increase = range_512 - random_y
                x_cut = int(original_x + x_increase)
                y_cut = int(original_y + y_increase)
                image_cut = original_image[0:x_cut, 0:y_cut]
                image_cut_size_x, image_cut_size_y = image_cut.shape

                x_zeroes = np.zeros((image_cut_size_x, range_512 - image_cut_size_y))
                y_zeroes = np.zeros((range_512 - image_cut_size_x, range_512))

                image_adjust1 = np.hstack((x_zeroes, image_cut))
                image_adjust = np.vstack((y_zeroes, image_adjust1))
                x_adjust = original_x + y_gap
                y_adjust = original_y + x_gap
            elif x_gap > 0 and y_gap < 0:  # 2
                x_increase = range_512 - random_x
                y_increase = random_y
                x_cut = int(original_x + x_increase)
                y_cut = int(original_y - y_increase)
                image_cut = original_image[0:x_cut, y_cut:range_512]
                image_cut_size_x, image_cut_size_y = image_cut.shape

                x_zeroes = np.zeros((image_cut_size_x, range_512 - image_cut_size_y))
                y_zeroes = np.zeros((range_512 - image_cut_size_x, range_512))

                image_adjust1 = np.hstack((image_cut, x_zeroes))
                image_adjust = np.vstack((y_zeroes, image_adjust1))
                x_adjust = original_x + y_gap  # 这个地方需要注意，加上的距离的轴相反
                y_adjust = original_y + x_gap  # 这个地方需要注意，加上的距离的轴相反
            elif x_gap < 0 and y_gap > 0:  # 3
                x_increase = random_x
                y_increase = range_512 - random_y
                x_cut = int(original_x - x_increase)
                y_cut = int(original_y + y_increase)
                image_cut = original_image[x_cut:range_512, 0:y_cut]
                image_cut_size_x, image_cut_size_y = image_cut.shape

                x_zeroes = np.zeros((image_cut_size_x, range_512 - image_cut_size_y))
                y_zeroes = np.zeros((range_512 - image_cut_size_x, range_512))

                image_adjust1 = np.hstack((x_zeroes, image_cut))
                image_adjust = np.vstack((image_adjust1, y_zeroes))
                x_adjust = original_x + y_gap
                y_adjust = original_y + x_gap
            elif x_gap < 0 and y_gap < 0:  # 4
                x_increase = random_x
                y_increase = random_y
                x_cut = int(original_x - x_increase)
                y_cut = int(original_y - y_increase)
                image_cut = original_image[x_cut:range_512, y_cut:range_512]
                image_cut_size_x, image_cut_size_y = image_cut.shape

                x_zeroes = np.zeros((image_cut_size_x, range_512 - image_cut_size_y))
                y_zeroes = np.zeros((range_512 - image_cut_size_x, range_512))

                image_adjust1 = np.hstack((image_cut, x_zeroes))
                image_adjust = np.vstack((image_adjust1, y_zeroes))
                x_adjust = original_x + y_gap
                y_adjust = original_y + x_gap

            new_images = image_adjust.reshape(range_512, range_512)
        except:
            continue
        solar_center_radius = np.array([x_adjust, y_adjust, solar_radius])
        new_X.append(new_images)
        new_Y.append(solar_center_radius)
    new_X = np.expand_dims(new_X, axis=-1)
    return np.array(new_X), np.array(new_Y)

def resize_solar_radius(X, Y): #scale
    width = 512
    height = 512
    shrink_r = 1 # 0.9
    enlarge_r = 1 # 1.1
    new_X = []
    new_Y = []
    for ic in range(len(Y[:, 0])):  # batch里面循环遍历
        shrink_enlarge = round(random.randint(0,1))
        original_image = X[ic, :, :, 0]
        original_x = Y[ic, 0]
        original_y = Y[ic, 1]
        solar_radius = Y[ic, 2]
        if shrink_enlarge == 0: #shrink
            r_factor = round(random.uniform(shrink_r, 1), 3)
            size = (int(width*r_factor), int(height*r_factor))
            r_X = cv2.resize(original_image, size)
            padd_value = round((width - size[0]) / 2)
            r_X = cv2.copyMakeBorder(r_X,padd_value,padd_value,padd_value,padd_value,cv2.BORDER_CONSTANT,value=[0])
            r_X = cv2.resize(r_X, [width, height])
            r_Y = np.array([original_x+padd_value, original_y+padd_value, solar_radius*r_factor])
        if shrink_enlarge == 1: #enlarge
            r_factor = round(random.uniform(enlarge_r, 1), 3)
            size = (int(width * r_factor), int(height * r_factor))
            padd_value = abs(round((width - size[0]) / 2))
            r_X = original_image[padd_value:height-padd_value, padd_value:width-padd_value]
            r_X = cv2.resize(r_X, [width, height])
            r_Y = np.array([original_x - padd_value, original_y - padd_value, solar_radius * r_factor])
        new_X.append(r_X)
        new_Y.append(r_Y)
    new_X = np.expand_dims(new_X, axis=-1)
    return np.array(new_X), np.array(new_Y)


# Build a data generator
def my_dataset_generator(batch_size,
                         train_dir,
                         labels_dir,
                         train_set,
                         training_rate,
                         is_binary_label=False):  # train set:True, valid set:False
    num_all_image = 7656
    range_512 = 512
    X_samples, Y_samples = load_ha_data(train_dir=train_dir, labels_dir=labels_dir, train_set=train_set,
                                        training_rate=training_rate)
    batch_num = int(len(X_samples) / batch_size)
    max_len = batch_num * batch_size
    X_samples = np.array(X_samples[:max_len])
    Y_samples = np.array(Y_samples[:max_len])

    X_batches = np.split(X_samples, batch_num)  # split according to batch size
    Y_batches = np.split(Y_samples, batch_num)

    i = 0  #
    ii = 0
    n = len(X_batches)
    while True:
        for b in range(len(X_batches)):  #
            i %= n
            X_batches[i], Y_batches[i] = shuffle_two_array(X_batches[i], Y_batches[i])  # 打乱每个batch数据
            X = np.array(list(map(load_batch_image, X_batches[i])))
            Y = np.array(Y_batches[i])
            # new_X, new_Y = resize_solar_radius(X, Y)
            new_X, new_Y = random_solardisk(X, Y)

            new_drawcicle_Y = new_Y
            try:
                if is_binary_label:  # 二进制标签
                    new_2_Y = []
                    for bi in range(batch_size):
                        new_tmp_Y = adjust_label_data(new_Y[bi,])
                        new_2_Y.append(new_tmp_Y)
                    new_Y = np.array(new_2_Y)

            except:
                continue

            try:
                X_new = np.empty(shape=X.shape)
                for c in np.arange(0, 0.1, 0.01):  #  Noise enhancement
                    for d in range(batch_size):  # Add noise according to the batch
                        t_tmp = new_X[d, :, :, 0] * 255.0
                        tt_tmp = t_tmp.astype(np.uint8)
                        X_1 = saltpepper_noise(tt_tmp, c)

                        X_1 = X_1 / 255.0
                        X_new[d, :, :, 0] = X_1
                        ii += 1
                    i += 1
                    draw_circle(X_new, new_drawcicle_Y, X_batches[i], batch_size)  # 画圆
                    yield X_new, new_Y  # you can comment this part when you don't need data augumentation.
            except:
                continue

            # yield new_X, new_Y ###  comment this when you need data augumentation.



# test generator
def testGenerator(test_path, target_size=(512, 512), as_gray=True):
    for file in sorted(os.listdir(test_path)):
        if (os.path.splitext(file)[-1][1:] == "png"):
            print(file)
            img = io.imread(os.path.join(test_path, file), as_gray=as_gray)
            img = img / 255.0

            img = trans.resize(img, target_size)
            img = np.reshape(img, (1,) + img.shape + (1,))
            yield img


# Save results
def saveResult(save_path, results, is_binary_label):
    if is_binary_label:
        range_512 = 512
        results_999 = []
        for ii in range(len(results[:, 0])):
            results_tmp = adjust_data_lable(results[ii, :])
            results_999.append(results_tmp)
        results_999 = np.array(results_999)
        np.savetxt(save_path + '/results/' + 'result.txt', results_999, '%f', delimiter=' ')
        i = 0
        for file in sorted(os.listdir(save_path)):
            if (os.path.splitext(file)[-1][1:] == "png"):
                img = io.imread(os.path.join(save_path, file), as_gray=True)
                cv2.circle(img, (results_999[i, 0], results_999[i, 1]), results_999[i, 2], (255, 255, 255), 2)
                cv2.imwrite(save_path + "/results/" + file, img)
                i += 1
    else:
        np.savetxt(save_path + '/results/' + 'result.txt', results, '%f', delimiter=' ')
        i = 0
        for file in sorted(os.listdir(save_path)):
            if (os.path.splitext(file)[-1][1:] == "png"):
                img = io.imread(os.path.join(save_path, file), as_gray=True)
                cv2.circle(img, (results[i, 0], results[i, 1]), results[i, 2], (255, 255, 255), 2)
                cv2.imwrite(save_path + "/results/" + file, img)
                i += 1

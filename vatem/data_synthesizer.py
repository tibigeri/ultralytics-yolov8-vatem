import os
import pickle
import random
import cv2
import copy
import shutil
from tqdm import tqdm



side_lines = {'vertical': [['mm1', 'mm2'], ['hkm1', 'hkm2'], ['fbm1', 'fbm2'], ['mem1', 'mem2']],
              'horizontal': [['fth1', 'fth2']],
              'single': ['th']}

top_lines = {'vertical': [['vszb1', 'vszbj2'], ['mszb1', 'mszj2'], ['hszb1', 'hszj2'], ['Ifszb1', 'Ifszj2'],['IIIfszb1', 'IIIfszj2']],
              'horizontal': [['fh1', 'fh2']],
              'single': []}

project_dir = os.getcwd()
sideImages_folder = r'dataset\original\sideImages'
topImages_folder = r'dataset\original\topImages'
sideMeasurements_pickle = r'dataset\original\sideImages\sideMeasurements.pickle'
topMeasurements_pickle = r'dataset\original\topImages\topMeasurements.pickle'

def get_side_measurements():
    pickle_path = os.path.join(project_dir, sideMeasurements_pickle)
    with open(pickle_path, 'rb') as f:
        return pickle.load(f)

def get_top_measurements():
    pickle_path = os.path.join(project_dir, topMeasurements_pickle)
    with open(pickle_path, 'rb') as f:
        return pickle.load(f)

def draw_points_and_show_im(img_name, img, points):
    im_height = img.shape[0]
    im_width = img.shape[1]
    for key, value in points.items():
        point_x = value[0]
        point_y = value[1]
        point_cord = (int(point_x * im_width), int(point_y * im_width)) # normalized by width
        cv2.putText(img, key, point_cord, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.circle(img, point_cord, 5, (0, 255, 0), -1)

    # Display the image
    cv2.imshow(img_name, img)


def visualize_random_images(num_images):
    # Load the image
    side_meas = synthesize_side_measurements()
    top_meas = synthesize_top_measurements()

    for _ in range(num_images):
        side_im_name, side_points = random.choice(list(side_meas.items()))
        top_im_name, top_points = random.choice(list(top_meas.items()))

        side_im = cv2.imread(os.path.join(project_dir, sideImages_folder, side_im_name))
        top_im = cv2.imread(os.path.join(project_dir, topImages_folder, top_im_name))

        draw_points_and_show_im(side_im_name, side_im, side_points)
        draw_points_and_show_im(top_im_name, top_im, top_points)

    cv2.waitKey(0)
    cv2.destroyAllWindows()



def synthesize_top_measurements():
    original_top_meas = get_top_measurements()
    synthesized_top_meas = copy.deepcopy(original_top_meas)
    for im_name, im_points in original_top_meas.items():

        faces_left = im_points['vszb1'][0] < im_points['fh2'][0]

        for vertical_line in top_lines['vertical']:
            left = vertical_line[0]
            right = vertical_line[1]
            if faces_left:
                if im_points[left][1] < im_points[right][1]:
                    synthesized_top_meas[im_name][left] = original_top_meas[im_name][right]
                    synthesized_top_meas[im_name][right] = original_top_meas[im_name][left]
            else:
                if im_points[left][1] > im_points[right][1]:
                    synthesized_top_meas[im_name][left] = original_top_meas[im_name][right]
                    synthesized_top_meas[im_name][right] = original_top_meas[im_name][left]


        for horizontal_line in top_lines['horizontal']:
            front = horizontal_line[0]
            back = horizontal_line[1]
            if faces_left:
                if im_points[front][0] > im_points[back][0]:
                    synthesized_top_meas[im_name][front] = original_top_meas[im_name][back]
                    synthesized_top_meas[im_name][back] = original_top_meas[im_name][front]
            else:
                if im_points[front][0] < im_points[back][0]:
                    synthesized_top_meas[im_name][front] = original_top_meas[im_name][back]
                    synthesized_top_meas[im_name][back] = original_top_meas[im_name][front]

    return synthesized_top_meas



def synthesize_side_measurements():
    original_side_meas = get_side_measurements()
    synthesized_side_meas = copy.deepcopy(original_side_meas)
    for im_name, im_points in original_side_meas.items():

        faces_left = im_points['mm1'][0] < im_points['th'][0]

        for vertical_line in side_lines['vertical']:
            bottom = vertical_line[0]
            top = vertical_line[1]
            if im_points[bottom][1] < im_points[top][1]:
                synthesized_side_meas[im_name][bottom] = original_side_meas[im_name][top]
                synthesized_side_meas[im_name][top] = original_side_meas[im_name][bottom]

        for horizontal_line in side_lines['horizontal']:
            front = horizontal_line[0]
            back = horizontal_line[1]

            if faces_left:
                if im_points[front][0] > im_points[back][0]:
                    synthesized_side_meas[im_name][front] = original_side_meas[im_name][back]
                    synthesized_side_meas[im_name][back] = original_side_meas[im_name][front]
            else:
                if im_points[front][0] < im_points[back][0]:
                    synthesized_side_meas[im_name][front] = original_side_meas[im_name][back]
                    synthesized_side_meas[im_name][back] = original_side_meas[im_name][front]

    return synthesized_side_meas



def create_cocoish_dataset():
    if not os.path.exists(os.path.join(project_dir, r'dataset\coco-vatem')):
        os.makedirs(os.path.join(project_dir, r'dataset\coco-vatem\side\images\train'))
        os.makedirs(os.path.join(project_dir, r'dataset\coco-vatem\side\images\val'))
        os.makedirs(os.path.join(project_dir, r'dataset\coco-vatem\side\labels\train'))
        os.makedirs(os.path.join(project_dir, r'dataset\coco-vatem\side\labels\val'))
        os.makedirs(os.path.join(project_dir, r'dataset\coco-vatem\top\images\train'))
        os.makedirs(os.path.join(project_dir, r'dataset\coco-vatem\top\images\val'))
        os.makedirs(os.path.join(project_dir, r'dataset\coco-vatem\top\labels\train'))
        os.makedirs(os.path.join(project_dir, r'dataset\coco-vatem\top\labels\val'))

    sides = synthesize_side_measurements()
    tops = synthesize_top_measurements()

    label_path_side = os.path.join(project_dir, r'dataset\coco-vatem\side\labels\train')
    label_path_top = os.path.join(project_dir, r'dataset\coco-vatem\top\labels\train')

    img_path_side = os.path.join(project_dir, r'dataset\coco-vatem\side\images\train')
    img_path_top = os.path.join(project_dir, r'dataset\coco-vatem\top\images\train')

    for (side_key, side_value) in tqdm(sides.items(), total=len(sides.items())):

        side_string = '19 0 0 0 0'

        for (side_kp_label, side_kp_coords)in side_value.items():
            side_string += f" {side_kp_coords[0]} {side_kp_coords[1]} 2"

        side_imname, _ = side_key.split('.')
        with open(label_path_side + '/' + side_imname + '.txt', 'w') as file:
            file.write(side_string)

        orig_side_img_path = os.path.join(project_dir, sideImages_folder) + f'/{side_key}'
        shutil.copy(orig_side_img_path, img_path_side)


    for (top_key, top_value) in tqdm(tops.items(), total=len(tops.items())):

        top_string = '19 0 0 0 0'

        for (top_kp_label, top_kp_coords) in top_value.items():
            top_string += f" {top_kp_coords[0]} {top_kp_coords[1]} 2"

        top_imname, _ = top_key.split('.')

        with open(label_path_top + '/' + top_imname + '.txt', 'w') as file:
            file.write(top_string)

        orig_top_img_path = os.path.join(project_dir, topImages_folder) + f'/{top_key}'

        shutil.copy(orig_top_img_path, img_path_top)

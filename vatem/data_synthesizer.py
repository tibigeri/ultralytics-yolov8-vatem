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

num_side_points = 11
num_top_points = 12

VALIDATION_RATIO = 0.2

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
        if key == 'bbox':
            center_x, center_y, width, height = value

            # NOTE: y paramters scaled with width of image
            x1 = (center_x - (width/2)) * im_width
            y1 = (center_y - (height/2)) * im_width
            x2 = (center_x + (width/2)) * im_width
            y2 = (center_y + (height/2)) * im_width

            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])

            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        else:
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

        bbox = get_bbox(im_points, faces_left)
        synthesized_top_meas[im_name]['bbox'] = bbox

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


        bbox = get_bbox(im_points, faces_left)
        synthesized_side_meas[im_name]['bbox'] = bbox

    return synthesized_side_meas


def get_bbox(keypoints, faces_left, include_head = True ,tol=0.05, tol_at_head = 0.4):

    if not include_head: tol_at_head = tol

    bbox_left = 0
    bbox_right = 0
    bbox_top = 0
    bbox_bottom = 0
    for kp_label, kp_coords in keypoints.items():
        x = kp_coords[0]
        y = kp_coords[1]
        if x < bbox_left or bbox_left == 0: bbox_left = x
        if x > bbox_right or bbox_right == 0: bbox_right = x
        if y < bbox_top or bbox_top == 0: bbox_top = y
        if y > bbox_bottom or bbox_bottom == 0: bbox_bottom = y

    height = (bbox_bottom - bbox_top)
    width = (bbox_right - bbox_left)
    bbox_left = bbox_left - (width * tol_at_head) if faces_left else bbox_left - (width * tol)
    bbox_right = bbox_right + (width * tol) if faces_left else bbox_right + (width * tol_at_head)
    bbox_top = bbox_top - (height * tol)
    bbox_bottom = bbox_bottom + (height * tol)

    #Format: xywh
    return [round((bbox_right + bbox_left) / 2 , 6), round((bbox_bottom + bbox_top) / 2 , 6),
            round((bbox_right - bbox_left) , 6), round((bbox_bottom - bbox_top) , 6)]

def create_cocoish_dataset():
    # Create dataset folder for cocoish dataset
    if os.path.exists(os.path.join(project_dir, r'dataset\coco-vatem')):
        shutil.rmtree(os.path.join(project_dir, r'dataset\coco-vatem'))

    os.makedirs(os.path.join(project_dir, r'dataset\coco-vatem\images\train'))
    os.makedirs(os.path.join(project_dir, r'dataset\coco-vatem\images\val'))
    os.makedirs(os.path.join(project_dir, r'dataset\coco-vatem\labels\train'))
    os.makedirs(os.path.join(project_dir, r'dataset\coco-vatem\labels\val'))

    # Get the synthesized raw data (flipped labels where needed)
    sides = synthesize_side_measurements()
    tops = synthesize_top_measurements()

    # Create variables for labels and images directories
    label_path = os.path.join(project_dir, r'dataset\coco-vatem\labels\train')
    img_path = os.path.join(project_dir, r'dataset\coco-vatem\images\train')

    val_label_path = os.path.join(project_dir, r'dataset\coco-vatem\labels\val')
    val_img_path = os.path.join(project_dir, r'dataset\coco-vatem\images\val')

    # Iterate over the raw data (side)
    for (im_name, kp_dict) in tqdm(sides.items(), total=len(sides.items())):

        side_im = cv2.imread(os.path.join(project_dir, sideImages_folder, im_name))
        fix_y_norm = side_im.shape[1]/side_im.shape[0] # in the raw dataset the y direction was also noramlized by the width of the image

        # First five numbers are: class id, bbox x, bbox y, bbox width, bbox height
        # Fix y norms in bbox coordinates
        label_data = f"0 {kp_dict['bbox'][0]} {round(kp_dict['bbox'][1] * fix_y_norm,6)} {kp_dict['bbox'][2]} {round(kp_dict['bbox'][3] * fix_y_norm,6)}"

        # Iterate over the keypoints dictionary of the image
        for (kp_label, kp_coords)in kp_dict.items():
            if kp_label != 'bbox':
                label_data += f" {kp_coords[0]} {round(kp_coords[1]*fix_y_norm,6)} 2.000000"

        # Append label_data with empty top keypoint coordinates
        for i in range(num_top_points):
            label_data += ' 0.000000 0.000000 0.000000'

        with open(label_path + '/' + im_name.split('.')[0] + '_s' + '.txt', 'w') as file:
            file.write(label_data)

        src_side_img_path = os.path.join(project_dir, sideImages_folder) + f'/{im_name}'
        dst_side_img_path = img_path + '/' + im_name.split('.')[0] + '_s.' + im_name.split('.')[1]
        shutil.copy(src_side_img_path, dst_side_img_path)

    # Iterate over the raw data (top)
    for (im_name, kp_dict) in tqdm(tops.items(), total=len(tops.items())):

        top_im = cv2.imread(os.path.join(project_dir, topImages_folder, im_name))
        fix_y_norm = top_im.shape[1]/top_im.shape[0] # in the raw dataset the y direction was also noramlized by the width of the image

        # First five numbers are: class id, bbox x, bbox y, bbox height, bbox width
        # Fix y norms in bbox coordinates
        label_data = f"0 {kp_dict['bbox'][0]} {round(kp_dict['bbox'][1]*fix_y_norm,6)} {kp_dict['bbox'][2]} {round(kp_dict['bbox'][3]*fix_y_norm,6)}"


        # Append label_data with empty side keypoint coordinates
        for i in range(num_side_points):
            label_data += ' 0.000000 0.000000 0.000000'

        # Iterate over the keypoints dictionary of the image
        for (kp_label, kp_coords) in kp_dict.items():
            if kp_label != 'bbox':
                label_data += f" {kp_coords[0]} {round(kp_coords[1]*fix_y_norm,6)} 2.000000"

        with open(label_path + '/' + im_name.split('.')[0]+ '_t' + '.txt', 'w') as file:
            file.write(label_data)

        src_top_img_path = os.path.join(project_dir, topImages_folder) + f'/{im_name}'
        dst_top_img_path = img_path + '/' + im_name.split('.')[0] + '_t.' + im_name.split('.')[1]
        shutil.copy(src_top_img_path, dst_top_img_path)

    # Move files to validation folder
    filenames = os.listdir(img_path)
    num_validation = int(len(filenames) * VALIDATION_RATIO)
    validation_filenames = random.sample(filenames, num_validation)

    print(f"Creating validation set ({VALIDATION_RATIO*100}%)...")
    for filename in validation_filenames:
        shutil.move(img_path + '/' + filename, val_img_path)
        shutil.move(label_path + '/' + filename.split('.')[0]+'.txt', val_label_path)


# Returns true if side, returns false if top
def is_side(file_name):
    name = file_name.split('.')[0]
    return name.split('_')[1] == 's'
#### Dependencies
from PIL import Image
from skimage.feature import greycomatrix, greycoprops

import matplotlib.pyplot as plt

#### Global variables
map_8bit_to_3bit = [i // 32 for i in range(256)]


#### Functions
def load_img(img_path):
    return Image.open(img_path).convert('L')

def load_preprocessed_img(img_path):
    img = load_img(img_path)
        
    return img

def get_img_size(img):
    return img.size

def get_img_width(img):
    return img.size[0]

def get_img_height(img):
    return img.size[1]

def print_img(img):
    plt.imshow(img, cmap='gray')
    
def get_resized_img(img, dimension):
    return img.resize(dimension)

def get_img_matrix(img, is_3bit_color=False):
    img_width, img_height = get_img_size(img)
    img_colors_list = list(img.getdata())
    
    color_id = 0
    img_matrix = []
    
    if(is_3bit_color):
        for row_id in range(img_height):
            temp_row = []
            for col_id in range(img_width):
                temp_row.append(map_8bit_to_3bit[img_colors_list[color_id]])

                color_id += 1
            img_matrix.append(temp_row)
    else:
        for row_id in range(img_height):
            temp_row = []
            for col_id in range(img_width):
                temp_row.append(img_colors_list[color_id])

                color_id += 1
            img_matrix.append(temp_row)
            
    return img_matrix
    
def get_img_features(img, glcm_components=['contrast', 'correlation', 'energy', 'homogeneity', 'ASM', 'dissimilarity'], distances=[1], angles=[0], levels=12, symmetric=False, normed=False):
    img_matrix = get_img_matrix(img, is_3bit_color=True)
    
    glcm_matrix = greycomatrix(img_matrix, distances, angles, levels, symmetric, normed)
    
    img_features = []
    for glcm_component in glcm_components:
        img_features.append(greycoprops(glcm_matrix, glcm_component)[0][0])
        
    return img_features

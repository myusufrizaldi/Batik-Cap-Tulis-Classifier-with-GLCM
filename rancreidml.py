#### Dependencies

import dec2bin as dtb
import imageglcm as iglcm

import glob
import math
import numpy as np
import os
import random
import statistics
import timeit

#### Class

class GlcmKnnClassifier:
    def __init__(self, model_name, k_neighbors=3, k_folds=5, glcm_components=['contrast', 'correlation', 'energy', 'homogeneity', 'ASM', 'dissimilarity']):
        self.model_name                 = model_name
        self.k_neighbors                = k_neighbors
        self.k_folds                    = k_folds
        self.perfect_test_overlap       = k_neighbors + (k_neighbors // 2)
        self.all_glcm_components        = ['contrast', 'correlation', 'energy', 'homogeneity', 'ASM', 'dissimilarity']
        self.all_glcm_components_length = len(self.all_glcm_components)
        self.glcm_components            = glcm_components
        self.glcm_components_length     = len(self.glcm_components)
        self.glcm_component_ids         = [self.all_glcm_components.index(glcm_component) for glcm_component in glcm_components]

        self.initialize()
        
    def initialize(self):
        self.validation_loss     = 0
        self.validation_accuracy = 0
        
        self.class_names     = []
        self.training_data   = []
        self.validation_data = []
        self.training_sample = []
        self.testing_sample  = []

        self.class_training_data_start_ids = []
        self.splitted_training_data = []

    def split_training_data(self):
        self.splitted_training_data = []

        class_training_data_start_ids_length = len(self.class_training_data_start_ids)
        for class_id in range(class_training_data_start_ids_length):
            if(class_id == class_training_data_start_ids_length - 1):
                end_constraint = len(self.training_data)
            else:
                end_constraint = self.class_training_data_start_ids[class_id + 1]

            class_training_data = self.training_data[self.class_training_data_start_ids[class_id]:end_constraint]

            splitted_class_training_data = []
            class_training_data_length = len(class_training_data)
            length_per_fold = int(class_training_data_length / self.k_folds)
            for slice_start_id in range(0, class_training_data_length, length_per_fold):
                if(slice_start_id == class_training_data_length - 1):
                    splitted_class_training_data.append(class_training_data[slice_start_id:])
                else:
                    splitted_class_training_data.append(class_training_data[slice_start_id:slice_start_id + length_per_fold])

            self.splitted_training_data.append(splitted_class_training_data)
        
    def load_data(self, training_path='training/', validation_path='validation/', data_path='data/', img_type='*.jpg', is_skip=False):
        self.initialize()
        
        print('Loading data...')
        self.class_names = [class_name for class_name in os.listdir(training_path)]
        
        training_data_path   = data_path + self.model_name + '_training.data'
        validation_data_path = data_path + self.model_name + '_validation.data'
        
        if(os.path.exists(training_data_path) and os.path.exists(validation_data_path) and not is_skip):
            training_data_str   = open(training_data_path, 'r').read().split('\n')
            validation_data_str = open(validation_data_path, 'r').read().split('\n')
            
            training_img_id = 0
            training_img_class = ''
            for row in training_data_str:
                row = row[1:len(row) - 1].split(', ')
                row[1:] =  [float(value) for value in row[1:]]
                self.training_data.append(tuple(row))

                if(not training_img_class == row[0]):
                    self.class_training_data_start_ids.append(training_img_id)
                    training_img_class = row[0]
                training_img_id += 1
            
            for row in validation_data_str:
                row = row[1:len(row) - 1].split(', ')
                row[1:] =  [float(value) for value in row[1:]]
                self.validation_data.append(tuple(row))
        else:           
            training_img_id = 0
            for class_name in self.class_names:
                self.class_training_data_start_ids.append(training_img_id)
                training_img_paths = glob.glob(training_path + class_name + '/' + img_type)

                for training_img_path in training_img_paths:
                    training_img = iglcm.load_preprocessed_img(training_img_path)
                    training_img_features = iglcm.get_img_features(training_img, self.all_glcm_components)
                    temp_row = [class_name] + training_img_features
                    self.training_data.append(tuple(temp_row))
                    
                    training_img_id += 1
                    
                validation_img_paths = glob.glob(validation_path + class_name + '/' + img_type)
                for validation_img_path in validation_img_paths:
                    validation_img = iglcm.load_preprocessed_img(validation_img_path)
                    validation_img_features = iglcm.get_img_features(validation_img, self.all_glcm_components)
                    temp_row = [class_name] + validation_img_features
                    self.validation_data.append(tuple(temp_row))
                    
            if(not os.path.exists(data_path)):
                os.mkdir(data_path)
                
            with open(training_data_path, 'w+') as file_writer:
                training_data_last_id = len(self.training_data) - 1
                for row_id in range(training_data_last_id):
                    file_writer.write(str(self.training_data[row_id]).replace('\'', '') + '\n')
                file_writer.write(str(self.training_data[training_data_last_id]).replace('\'', ''))
                
            with open(validation_data_path, 'w+') as file_writer:
                validation_data_last_id = len(self.validation_data) - 1
                for row_id in range(validation_data_last_id):
                    file_writer.write(str(self.validation_data[row_id]).replace('\'', '') + '\n')
                file_writer.write(str(self.validation_data[row_id]).replace('\'', ''))
                
        self.split_training_data()

        training_data_length   = len(self.training_data)
        validation_data_length = len(self.validation_data)
        class_names_length     = len(self.class_names)
                
        print('--> Done (' + str(training_data_length) + ' training images and ' + str(validation_data_length) + ' validation images, into ' + str(class_names_length) + ' class).\n')
        
    def get_euclidean_distance(self, img_features_1, img_features_2):
        euclidean_distance = 0.0
        
        for glcm_id in range(self.all_glcm_components_length):
            if(glcm_id in self.glcm_component_ids):
                euclidean_distance += (img_features_1[glcm_id] - img_features_2[glcm_id]) ** 2
            
        return math.sqrt(euclidean_distance)
        
    def get_img_features_class(self, img_features):
        minimum_euclidean_distances = [-1 for i in range(self.k_neighbors)]
        minimum_img_class_names = ['unknown' for i in range(self.k_neighbors)]
        
        is_first_loop = True
        for row in self.training_sample:
            training_sample_img_class_name = row[0]
            training_sample_img_features   = row[1:]
            
            euclidean_distance = self.get_euclidean_distance(training_sample_img_features, img_features)
            
            if(is_first_loop):
                minimum_euclidean_distances[0] = euclidean_distance
                minimum_img_class_names[0]     = training_sample_img_class_name
                is_first_loop = False
            else:
                for i in range(self.k_neighbors):
                    if(euclidean_distance < minimum_euclidean_distances[i] or minimum_euclidean_distances[i] == -1):
                        for j in range(self.k_neighbors - 1, i, -1):
                            minimum_euclidean_distances[j] = minimum_euclidean_distances[j - 1]
                            minimum_img_class_names[j] = minimum_img_class_names[j - 1]
                        minimum_euclidean_distances[i] = euclidean_distance
                        minimum_img_class_names[i] = training_sample_img_class_name
                        break
                        
        minimum_img_class_names = list(filter(lambda val: val != 'unknown', minimum_img_class_names))
        img_class_name = max(set(minimum_img_class_names), key=minimum_img_class_names.count)
        
        return img_class_name
    
    def deep_train(self, training_rate=0.8, epochs=10, is_accuracy_oriented=False):
        print('Deep training has been started. It will take a lot of time, so take your time :)')       
        
        best_validation_accuracy = 0
        best_validation_accuracy_training_sample = []
        best_validation_accuracy_testing_sample = []
        best_validation_accuracy_glcm_components = []
        
        subsets_length_constraint = 2 ** self.all_glcm_components_length
        for subset_num in range(1, subsets_length_constraint):
            temp_glcm_components = []
            
            reversed_binary_num_str = dtb.get_reversed_binary_num_as_string(subset_num)
            reversed_binary_num_str_length = len(reversed_binary_num_str)
            for glcm_id in range(reversed_binary_num_str_length):
                if(reversed_binary_num_str[glcm_id] == '1'):
                    temp_glcm_components.append(self.all_glcm_components[glcm_id])
                    
            self.set_glcm_components(temp_glcm_components)
            
            print('Training ' + str(subset_num) + '/' + str(subsets_length_constraint - 1))
            print('Training on GLCM components: ' + str(temp_glcm_components))
            temp_validation_accuracy, temp_validation_loss = self.train(training_rate, epochs, is_accuracy_oriented)
            
            if(temp_validation_accuracy > best_validation_accuracy):
                best_validation_accuracy = temp_validation_accuracy
                best_validation_accuracy_glcm_components = temp_glcm_components
                best_validation_accuracy_training_sample = self.training_sample
                best_validation_accuracy_testing_sample  = self.testing_sample
                
        print('\n')
        print('Deep training completed.\n')
        print('Deep training report:')
        print('Best val_acc                      : ' + str(best_validation_accuracy))
        print('GLCM components for best val_acc  : ' + str(best_validation_accuracy_glcm_components))
        
        self.training_sample = best_validation_accuracy_training_sample
        self.testing_sample  = best_validation_accuracy_testing_sample
        self.set_glcm_components(temp_glcm_components)

        if(is_accuracy_oriented):
            print('\nYour model has been updated with best accuracy model.')
        else:
            print('\nYour model has been updated with best stablility model.')
        
        return best_validation_accuracy_glcm_components
        
    def train(self, training_rate=0.8, epochs=20, is_accuracy_oriented=False):
        print('Training...')
        training_sample_history = []
        testing_sample_history   = []
        self.training_sample    = []
        self.testing_sample     = []
        
        max_validation_accuracy = -1
        min_validation_loss     = 100
        max_validation_accuracy_sample_id = -1
        validation_accuracy, validation_loss = 0, 0
        
        perfect_test_count = 0
        for epoch in range(epochs):
            print('    Epoch ' + str(epoch + 1) + '/' + str(epochs))
            epoch_start_time = timeit.default_timer()

            training_k_folds = int(training_rate * self.k_folds)
            if(training_k_folds == 0):
                training_k_folds = 1
            testing_k_folds  = self.k_folds - training_k_folds

            available_folds = [i for i in range(self.k_folds)]
            splitted_training_data_length = len(self.splitted_training_data)
            for i in range(testing_k_folds):
                random_fold_id = random.choice(available_folds)
                available_folds.remove(random_fold_id)

                for class_id in range(splitted_training_data_length):
                    self.testing_sample += self.splitted_training_data[class_id][random_fold_id]

            for fold_id in available_folds:
                for class_id in range(splitted_training_data_length):
                    self.training_sample += self.splitted_training_data[class_id][fold_id]

            training_sample_history.append(self.training_sample)
            testing_sample_history.append(self.testing_sample)
                    
            testing_accuracy, testing_loss = self.test()
            validation_accuracy, validation_loss = self.validate()
            
            epoch_end_time = timeit.default_timer()
            
            epoch_time               = '{:.4f}'.format(round(epoch_end_time - epoch_start_time, 4)) + ' s'
            testing_accuracy_str     = '{:.4f}'.format(testing_accuracy)
            testing_loss_str         = '{:.4f}'.format(testing_loss)
            validation_accuracy_str  = '{:.4f}'.format(validation_accuracy)
            validation_loss_str      = '{:.4f}'.format(validation_loss)
            
            print('    --> ' + 'time: ' + epoch_time + ' - test_loss: ' + testing_loss_str + ' - test_acc: ' + testing_accuracy_str + ' - val_loss: ' + validation_loss_str + ' - val_acc: ' + validation_accuracy_str)        
        
            if(not is_accuracy_oriented and testing_accuracy == 1.0):
                perfect_test_count += 1
                if(perfect_test_count == self.perfect_test_overlap):
                    print('Epochs end, the perfect test overlap has been reached.')
                    break
            else:
                if(validation_accuracy > max_validation_accuracy):
                    max_validation_accuracy = validation_accuracy
                    min_validation_loss     = validation_loss
                    max_validation_accuracy_sample_id = epoch
                    
        self.validation_accuracy = validation_accuracy
        self.validation_loss     = validation_loss

        if(is_accuracy_oriented):
            validation_accuracy_str  = '{:.4f}'.format(max_validation_accuracy)
            validation_loss_str      = '{:.4f}'.format(min_validation_loss)

            self.training_sample = training_sample_history[max_validation_accuracy_sample_id]
            self.testing_sample  = testing_sample_history[max_validation_accuracy_sample_id]
            print('--> Done, loss: ' + validation_loss_str + ' - acc: ' + validation_accuracy_str + '\n')
            return max_validation_accuracy, min_validation_loss
        else:
            print('--> Done, loss: ' + validation_loss_str + ' - acc: ' + validation_accuracy_str + '\n')
            return validation_accuracy, validation_loss
        
    def test(self):
        total_correct_answer = 0
        total_guess = 0
        
        for row in self.testing_sample:
            expected_testing_img_class_name = row[0]
            testing_img_features = row[1:]
            
            testing_img_class_name = self.get_img_features_class(testing_img_features)
            
            if(expected_testing_img_class_name == testing_img_class_name):
                total_correct_answer += 1
                
            total_guess += 1
            
        testing_accuracy = round(total_correct_answer / total_guess, 4)
        testing_loss = 1 - testing_accuracy
        
        return testing_accuracy, testing_loss
    
    def validate(self):
        total_correct_answer = 0
        total_guess = 0
        
        for row in self.validation_data:
            expected_validation_img_class_name = row[0]
            validation_img_features = row[1:]
            
            validation_img_class_name = self.get_img_features_class(validation_img_features)
            
            if(expected_validation_img_class_name == validation_img_class_name):
                total_correct_answer += 1
                
            total_guess += 1
            
        validation_accuracy = round(total_correct_answer / total_guess, 4)
        validation_loss = 1 - validation_accuracy
        
        return validation_accuracy, validation_loss

    def set_properties(self, model_name, k_neighbors, glcm_components):
        self.model_name = model_name
        self.k_neighbors = k_neighbors
        self.glcm_components = glcm_components
        
    def set_model_name(self, model_name):
        self.model_name = model_name
        
    def set_k_neighbors(self, k_neighbors):
        self.k_neighbors = k_neighbors
        
    def set_glcm_components(self, glcm_components):
        self.glcm_components    = glcm_components
        self.glcm_component_ids = [self.all_glcm_components.index(glcm_component) for glcm_component in glcm_components]
####  Local Dependencies

import dec2bin as dtb
import imageglcm as iglcm

####  Global Dependencies

import glob
import json
import math
import numpy as np
import os
import pandas as pd
import random
import statistics
import timeit



####  Classification Methods Class

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
        self.validation_loss      = 0
        self.validation_accuracy  = 0
        self.validation_precision = 0
        self.validation_recall    = 0
        self.validation_f1_score  = 0

        self.validation_precisions = []
        self.validation_recalls    = []
        self.validation_f1_scores  = []

        self.validation_metrics             = pd.DataFrame(columns=['class', 'precision', 'recall', 'f1-score'])
        self.consecutive_validation_metrics = {}
        
        self.class_names     = []
        self.training_data   = []
        self.validation_data = []
        self.training_sample = {}
        self.testing_sample  = {}

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
        print('')
        
    def get_euclidean_distance(self, img_features_1, img_features_2):
        euclidean_distance = 0.0
        
        for glcm_id in range(self.all_glcm_components_length):
            if(glcm_id in self.glcm_component_ids):
                euclidean_distance += (img_features_1[glcm_id] - img_features_2[glcm_id]) ** 2
            
        return math.sqrt(euclidean_distance)
        
    def get_img_features_class(self, img_features):
        minimum_euclidean_distances = [99999999 for i in range(self.k_neighbors)]
        minimum_img_class_names = ['unknown' for i in range(self.k_neighbors)]
        
        for training_sample_img_features in self.training_sample:
            training_sample_img_class_name = self.training_sample[training_sample_img_features][0]
            training_sample_img_count      = self.training_sample[training_sample_img_features][1]

            euclidean_distance = self.get_euclidean_distance(training_sample_img_features, img_features)

            for i in range(training_sample_img_count):
                if(i == self.k_neighbors):
                    break

                # Thanks to Allah, then my friend github.com/alien087 (Dino Febriyanto) on this idea, slightly faster
                maximum_euclidean_distance    = max(minimum_euclidean_distances)
                maximum_euclidean_distance_id = minimum_euclidean_distances.index(maximum_euclidean_distance)
                if(euclidean_distance < maximum_euclidean_distance):
                    minimum_euclidean_distances[maximum_euclidean_distance_id] = euclidean_distance
                    minimum_img_class_names[maximum_euclidean_distance_id] = training_sample_img_class_name
                else:
                    break

        minimum_img_class_names = list(filter(lambda val: val != 'unknown', minimum_img_class_names))
        img_class_name = max(set(minimum_img_class_names), key=minimum_img_class_names.count)
        
        return img_class_name

    def deep_train(self, k_neighbors_list=[3], training_rate_list=[0.8], iterations_list=[5], epochs_list=[10], is_accuracy_oriented=False, is_full_epochs=True, is_skip_epoch_validation=False):
        print('Deep training has been started. It will take a lot of time, so take your time:)\n')
        print('')

        self.consecutive_validation_metrics = {}

        deep_train_time = 0
        training_length = len(epochs_list)
        for training_id in range(training_length):
            print('Training ' + str(training_id + 1) + '/' + str(training_length) + ' with k = ' + str(k_neighbors_list[training_id]) + ', ' + str(training_rate_list[training_id]) + ' training rate, ' + str(iterations_list[training_id]) + ' iterations, and ' + str(epochs_list[training_id]) + ' epochs')
            
            training_time = 0
            training_name = str(training_id + 1) + '_k' + str(k_neighbors_list[training_id]) + '_' + str(training_rate_list[training_id]) + 'tr_' + str(iterations_list[training_id]) + 'iters_' + str(epochs_list[training_id]) + 'eps'
            self.k_neighbors = k_neighbors_list[training_id]
            self.consecutive_validation_metrics[training_name] = []
            for iteration in range(iterations_list[training_id]):
                print('Iteration ' + str(iteration + 1) + '/' + str(iterations_list[training_id]))

                iteration_start_time = timeit.default_timer()

                self.train(training_rate_list[training_id], epochs_list[training_id], is_accuracy_oriented, is_full_epochs, is_skip_epoch_validation)
                
                iteration_end_time   = timeit.default_timer()
                iteration_time       = iteration_end_time - iteration_start_time
                print('Iteration done, time elapsed: ' + self.get_formatted_time(iteration_time) + '\n')
                print()

                training_time += iteration_time

                self.consecutive_validation_metrics[training_name].append(self.validation_metrics)
                self.save_validation_metrics(training_rate_list[training_id], epochs_list[training_id], iteration + 1)

            deep_train_time += training_time

            print('Training done, time elapsed: ' + self.get_formatted_time(training_time) + '\n')
            print()

        print('Deep training done, time elapsed: ' + self.get_formatted_time(deep_train_time) + '\n')
        print()

        self.save_consecutive_validation_metrics()

    def wide_train(self, training_rate=0.8, epochs=10, is_accuracy_oriented=False):
        print('Wide training has been started. It will take a lot of time, so take your time :)\n')
        print('\n')       
        
        best_validation_accuracy = 0
        best_validation_accuracy_training_sample = {}
        best_validation_accuracy_testing_sample = {}
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
        print('Wide training completed.\n')
        print('Wide training report:')
        print('Best val_acc                      : ' + str(best_validation_accuracy))
        print('GLCM components for best val_acc  : ' + str(best_validation_accuracy_glcm_components))
        
        self.training_sample = best_validation_accuracy_training_sample
        self.testing_sample  = best_validation_accuracy_testing_sample
        self.set_glcm_components(temp_glcm_components)
        self.save_validation_metrics(training_rate, epochs, 'best')

        if(is_accuracy_oriented):
            print('\nYour model has been updated with best accuracy model.')
        else:
            print('\nYour model has been updated with best stablility model.')
        
        return best_validation_accuracy_glcm_components
        
    def train(self, training_rate=0.8, epochs=20, is_accuracy_oriented=False, is_full_epochs=False, is_skip_epoch_validation=False):
        print('Training...')
        self.training_sample       = {}
        self.testing_sample        = {}
        training_sample_history    = []
        testing_sample_history     = []
        validation_metrics_history = []
        
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
                    for training_data_row in self.splitted_training_data[class_id][random_fold_id]:
                        training_data_class_name   = training_data_row[0]
                        training_data_row_features = tuple(training_data_row[1:])

                        if(training_data_row_features in self.testing_sample):
                            self.testing_sample[training_data_row_features][1] += 1
                        else:
                            self.testing_sample[training_data_row_features] = [training_data_class_name, 1]

            for fold_id in available_folds:
                for class_id in range(splitted_training_data_length):
                    for training_data_row in self.splitted_training_data[class_id][fold_id]:
                        training_data_class_name   = training_data_row[0]
                        training_data_row_features = tuple(training_data_row[1:])

                        if(training_data_row_features in self.training_sample):
                            self.training_sample[training_data_row_features][1] += 1
                        else:
                            self.training_sample[training_data_row_features] = [training_data_class_name, 1]

            training_sample_history.append(self.training_sample)
            testing_sample_history.append(self.testing_sample)
                    
            testing_accuracy, testing_loss = self.test()
            
            epoch_end_time = timeit.default_timer()
            
            epoch_time               = '{:.4f}'.format(round(epoch_end_time - epoch_start_time, 4)) + ' s'
            testing_accuracy_str     = '{:.4f}'.format(testing_accuracy)
            testing_loss_str         = '{:.4f}'.format(testing_loss)

            if(not is_skip_epoch_validation):
                validation_accuracy, validation_loss = self.validate()
                validation_accuracy_str  = '{:.4f}'.format(validation_accuracy)
                validation_loss_str      = '{:.4f}'.format(validation_loss)

                print('    --> ' + 'time: ' + epoch_time + ' - test_loss: ' + testing_loss_str + ' - test_acc: ' + testing_accuracy_str + ' - val_loss: ' + validation_loss_str + ' - val_acc: ' + validation_accuracy_str)        
            else:
                print('    --> ' + 'time: ' + epoch_time + ' - test_loss: ' + testing_loss_str + ' - test_acc: ' + testing_accuracy_str)   

            if(is_accuracy_oriented):
                validation_metrics_history.append(self.validation_metrics)

                if(validation_accuracy > max_validation_accuracy):
                    max_validation_accuracy = validation_accuracy
                    min_validation_loss     = validation_loss
                    max_validation_accuracy_sample_id = epoch                
            else:
                if(testing_accuracy == 1.0):
                    perfect_test_count += 1
                    if(perfect_test_count == self.perfect_test_overlap and not is_full_epochs):
                        print('Epochs end, the perfect test overlap has been reached.')
                        break
                

        if(is_skip_epoch_validation):
            self.validate()
        else:
            self.validation_accuracy = validation_accuracy
            self.validation_loss     = validation_loss

        if(is_accuracy_oriented):
            validation_accuracy_str  = '{:.4f}'.format(max_validation_accuracy)
            validation_loss_str      = '{:.4f}'.format(min_validation_loss)

            self.training_sample     = training_sample_history[max_validation_accuracy_sample_id]
            self.testing_sample      = testing_sample_history[max_validation_accuracy_sample_id]
            self.validation_metrics  = validation_metrics_history[max_validation_accuracy_sample_id]
            self.validation_accuracy = max_validation_accuracy
            self.validation_loss     = min_validation_loss
            print('--> Done, loss: ' + validation_loss_str + ' - acc: ' + validation_accuracy_str + '\n')
            self.print_validation_metrics()
            return max_validation_accuracy, min_validation_loss
        else:
            validation_accuracy_str  = '{:.4f}'.format(self.validation_accuracy)
            validation_loss_str      = '{:.4f}'.format(self.validation_loss)

            print('--> Done, loss: ' + validation_loss_str + ' - acc: ' + validation_accuracy_str + '\n')
            self.print_validation_metrics()
            return validation_accuracy, validation_loss
        
    def test(self):
        total_correct_answer = 0
        total_guess = 0
        
        for testing_img_features in self.testing_sample:
            testing_img_properties = self.testing_sample[testing_img_features]
            expected_testing_img_class_name = testing_img_properties[0]
            testing_img_count = testing_img_properties[1]

            testing_img_class_name_prediction = self.get_img_features_class(testing_img_features)

            if(expected_testing_img_class_name == testing_img_class_name_prediction):
                total_correct_answer += testing_img_count
                
            total_guess += testing_img_count
            
        testing_accuracy = round(total_correct_answer / total_guess, 4)
        testing_loss = 1 - testing_accuracy
        
        return testing_accuracy, testing_loss
    
    def validate(self):
        class_names_length = len(self.class_names)
        true_guess     = [0 for i in range(class_names_length)]
        false_guess    = [0 for i in range(class_names_length)]

        for row in self.validation_data:
            expected_validation_img_class_name = row[0]
            validation_img_features = row[1:]
            
            validation_img_class_name = self.get_img_features_class(validation_img_features)
            
            if(expected_validation_img_class_name == validation_img_class_name):
                true_guess[self.class_names.index(validation_img_class_name)] += 1
            else:
                false_guess[self.class_names.index(validation_img_class_name)] += 1            
        
        self.validation_precisions = []
        self.validation_recalls    = []
        self.validation_f1_scores  = []
        
        total_correct_answer = 0
        total_guess = 0
        for class_id in range(class_names_length):
            total_correct_answer += true_guess[class_id]
            total_guess += true_guess[class_id] + false_guess[class_id]

            class_precision = round(true_guess[class_id] / (true_guess[class_id] + sum(false_guess[0:class_id] + false_guess[class_id+1:])), 4)
            class_recall    = round(true_guess[class_id] / (true_guess[class_id] + false_guess[class_id]), 4)
            class_f1_score  = round((2 * class_precision * class_recall) / (class_precision + class_recall), 4)

            self.validation_precisions.append(class_precision)
            self.validation_recalls.append(class_recall)
            self.validation_f1_scores.append(class_f1_score)

        self.validation_accuracy  = round(total_correct_answer / total_guess, 4)
        self.validation_loss      = 1 - self.validation_accuracy
        self.validation_precision = round(sum(self.validation_precisions) / len(self.validation_precisions), 4)
        self.validation_recall    = round(sum(self.validation_recalls) / len(self.validation_recalls), 4)
        self.validation_f1_score  = round(sum(self.validation_f1_scores) / len(self.validation_f1_scores), 4)

        self.validation_metrics = pd.DataFrame(columns=['class', 'precision', 'recall', 'f1-score'])
        for class_id in range(len(self.class_names)):
            self.validation_metrics.loc[len(self.validation_metrics)] = [self.class_names[class_id], self.validation_precisions[class_id], self.validation_recalls[class_id], self.validation_f1_scores[class_id]]
        self.validation_metrics.loc[len(self.validation_metrics)] = ['[avg]', self.validation_precision, self.validation_recall, self.validation_f1_score]

        return self.validation_accuracy, self.validation_loss

    def predict(self, img):
        img_features = iglcm.get_img_features(img)
        img_class_name = self.get_img_features_class(img_features)

        return img_class_name

    def print_validation_metrics(self):
        print(self.validation_metrics)
        print()

    def save_validation_metrics(self, training_rate, epochs, iteration):
        validation_metrics_path = 'validation_metrics/'
        if(not os.path.exists(validation_metrics_path)):
            os.mkdir(validation_metrics_path)
        validation_metrics_path = validation_metrics_path + self.model_name + '_' + str(training_rate) + 'tr_' + str(epochs) + 'eps_' + str(iteration) + '.csv'
        self.validation_metrics.to_csv(validation_metrics_path)

    def save_consecutive_validation_metrics(self):
        validation_metrics_path = 'validation_metrics/'
        excel_writer = pd.ExcelWriter(validation_metrics_path + self.model_name + '_deep_training_metrics.xlsx')
        
        for training_name in self.consecutive_validation_metrics:
            temp_sheet = pd.DataFrame(columns=['class', 'precision', 'recall', 'f1-score'])

            for iteration_metrics in self.consecutive_validation_metrics[training_name]:
                for row_id, row in iteration_metrics.iloc[:].iterrows():
                    temp_sheet.loc[len(temp_sheet)] = row
                temp_sheet.loc[len(temp_sheet)] = ['', '', '', '']
            
            temp_sheet = temp_sheet.head(-1)
            temp_sheet.to_excel(excel_writer, training_name)

        excel_writer.save()
        print('Your consecutive validation metrics has been saved.\n')

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

    def get_data_as_str(self, data):
        data_str = ''
        data_last_id = len(data) - 1
        for row_id in range(data_last_id):
            data_str += str(data[row_id]).replace('\'', '') + '\n'
        data_str += str(data[data_last_id]).replace('\'', '')

        return data_str

    def get_data_as_list(self, data_str):
        data_list = []
        data_str_list = data_str.split('\n')
        for row in data_str_list:
            row = row[1:len(row) - 1].split(', ')
            row[1:] = [float(value) for value in row[1:]]
            data_list.append(tuple(row))

        return data_list

    def get_sample_as_str(self, sample):
        sample_str = ''

        for sample_img_features in sample:
            sample_img_properties = sample[sample_img_features]
            sample_str += str((sample_img_properties[0], sample_img_properties[1], str(sample_img_features).replace('(', '').replace(')', ''))).replace('\'', '') + '\n'
        sample_str = sample_str[:len(sample_str) - 2]

        return sample_str

    def get_sample_as_dict(self, sample_str):
        sample_dict = {}
        sample_str_list = sample_str.split('\n')
        for row in sample_str_list:
            row = row[1:len(row) - 1].split(', ')

            img_class_name = row[0]
            img_count = row[1]
            img_features = tuple([float(value) for value in row[2:]])

            sample_dict[img_features] = tuple(img_class_name, img_count)

        return sample_dict

    def save(self):
        model_json = {}
        model_json['model_name'] = self.model_name
        model_json['k_neighbors'] = self.k_neighbors
        model_json['k_folds'] = self.k_folds
        model_json['perfect_test_overlap'] = self.perfect_test_overlap
        model_json['all_glcm_components'] = self.all_glcm_components
        model_json['glcm_components'] = self.glcm_components
        model_json['glcm_component_ids'] = [self.all_glcm_components.index(glcm_component) for glcm_component in self.glcm_components]
        model_json['validation_loss'] = self.validation_loss
        model_json['validation_accuracy'] = self.validation_accuracy
        model_json['class_names'] = self.class_names
        model_json['training_data'] = self.get_data_as_str(self.training_data)
        model_json['validation_data'] = self.get_data_as_str(self.validation_data)
        model_json['training_sample'] = self.get_sample_as_str(self.training_sample)
        model_json['testing_sample'] = self.get_sample_as_str(self.testing_sample)
        model_json['class_training_data_start_ids'] = self.class_training_data_start_ids
        

        model_json_path = self.model_name + '.json'
        with open(model_json_path, 'w+') as file_writer:
            json.dump(model_json, file_writer)

        print('Your model has been saved.\n')

    def load(self):
        self.initialize()

        model_json_path = self.model_name + '.json'
        with open(model_json_path) as file_reader:
            model_data = json.load(file_reader)
            
            self.model_name = model_data['model_name']
            self.k_neighbors = model_data['k_neighbors']
            self.k_folds = model_data['k_folds']
            self.perfect_test_overlap = model_data['perfect_test_overlap']
            self.all_glcm_components = model_data['all_glcm_components']
            self.all_glcm_components_length = len(self.all_glcm_components)
            self.glcm_components = model_data['glcm_components']
            self.glcm_components_length = len(self.glcm_components)
            self.glcm_component_ids = model_data['glcm_component_ids']

            self.validation_loss = model_data['validation_loss']
            self.validation_accuracy = model_data['validation_accuracy']
            self.class_names = model_data['class_names']
            self.training_data = self.get_data_as_list(model_data['training_data'])
            self.validation_data = self.get_data_as_list(model_data['validation_data'])
            self.training_sample = self.get_sample_as_dict(model_data['training_sample'])
            self.testing_sample = self.get_sample_as_dict(model_data['testing_sample'])
            self.class_training_data_start_ids = model_data['class_training_data_start_ids']
            self.split_training_data()

        print('Your model has been loaded with acc: ' + str(self.validation_accuracy) + ', all set, you\'re ready to predict!\n')
        
    def get_formatted_time(self, time_in_seconds):
        formatted_time = ''
        hours   = int(time_in_seconds // 3600)
        time_in_seconds %= 3600
        minutes = int(time_in_seconds // 60)
        seconds = int(time_in_seconds % 60)

        if(hours > 0):
            formatted_time += str(hours) + ' hours '
        if(minutes > 0):
            formatted_time += str(minutes) + ' mins '
        if(not (seconds == 0 and (hours > 0 or minutes > 0))):
            formatted_time += str(seconds) + ' secs'

        return formatted_time
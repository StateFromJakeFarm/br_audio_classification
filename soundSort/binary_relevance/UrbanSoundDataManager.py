import os
import torch
import random
import librosa
import numpy as np

class UrbanSoundDataManager:
    '''
    Load files from UrbanSound8K dataset for training and testing
    '''
    def __init__(self, audio_dir, test_fold=None):
        '''
        Constructor
        '''
        self.audio_dir = audio_dir

        # Store dataset classes as dictionary
        self.classes = {
            0: 'air_conditioner',
            1: 'car_horn',
            2: 'children_playing',
            3: 'dog_bark',
            4: 'drilling',
            5: 'engine_idling',
            6: 'gun_shot',
            7: 'jackhammer',
            8: 'siren',
            9: 'street_music'
        }

        # Determine which fold will be used as test set; all others used for training
        if test_fold is None:
            self.test_fold = random.randint(1, 10)
        elif 1 <= test_fold and test_fold <= 10:
            self.test_fold = test_fold
        else:
            raise ValueError('test_fold must be in range [1, 10]')

        # Compile list of all training file paths
        self.train_files = []
        for fold_num in range(1, 11):
            if fold_num == self.test_fold:
                # Don't add testing samples to training set
                continue

            fold = 'fold' + str(fold_num)
            fold_path = os.path.join(audio_dir, fold)
            for f_name in os.listdir(fold_path):
                self.train_files.append(os.path.join(fold_path, f_name))

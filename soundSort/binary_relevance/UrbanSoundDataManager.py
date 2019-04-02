import os
import torch
import random
import librosa
import numpy as np

class UrbanSoundDataManager:
    '''
    Load files from UrbanSound8K dataset for training and testing
    '''
    def __init__(self, audio_dir, test_fold=None, sr=8000, file_duration=4, chunk_duration=0.1):
        '''
        Constructor
        '''
        self.audio_dir = audio_dir
        self.sr = sr

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

        # Compile lists of all training and testing file paths
        self.train_files = []
        self.test_files = []
        for fold_num in range(1, 11):
            fold = 'fold' + str(fold_num)
            fold_path = os.path.join(audio_dir, fold)
            for f_name in os.listdir(fold_path):
                f_path = os.path.join(fold_path, f_name)

                if fold_num == self.test_fold:
                    # Belongs in testing set
                    self.test_files.append(f_path)
                else:
                    # Belongs in training set
                    self.train_files.append(f_path)

        # Shuffle training set order
        random.shuffle(self.train_files)

        # Batch size details
        self.file_duration = file_duration
        self.chunks = int(file_duration/chunk_duration)
        self.chunk_len = int(chunk_duration*self.sr)
        self.total_samples = self.sr*file_duration

        # Iterators to keep track of where we are in the training and testing sets
        self.i_train = 0
        self.i_test = 0

    def get_batch(self, type, size=10):
        '''
        Get next batch of shape (batch, seq_len, seq), which is representative
        of (file, chunks, chunk_len)
        '''
        # Gross hack to actually increment the iterator for the set being used
        if type == 'train':
            file_set = self.train_files
            iterator = self.i_train

            # Increment iterator
            self.i_train += size
            if self.i_train + size >= len(self.train_files):
                self.i_train = 0
        else:
            file_set = self.test_files
            iterator = self.i_test

            # Increment iterator
            self.i_test += size
            if self.i_test + size >= len(self.test_files):
                self.i_test = 0

        # Compile file data for this chunk into tensor
        batch = np.zeros((size, self.chunks, self.chunk_len), dtype=float)
        labels = []
        for i, file in enumerate(file_set[iterator:iterator+size]):
            # Extract label
            labels.append(int(os.path.split(file)[-1].split('-')[1]))

            # Load data
            Y, sr = librosa.core.load(file, sr=self.sr, duration=self.file_duration)

            if Y.shape[0] < self.total_samples:
                # Pad this array with zeros on the end
                Y = np.pad(Y, (0, self.total_samples-Y.shape[0]), 'constant')

            # Chunk-up data
            for chunk in range(self.chunks):
                batch[i][chunk] = Y[chunk*self.chunk_len:chunk*self.chunk_len+self.chunk_len]

        return torch.from_numpy(batch).float(), labels

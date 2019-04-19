import os
import torch
import random
import librosa
import numpy as np

from numpy.fft import fft

class UrbanSoundDataManager:
    '''
    Load files from UrbanSound8K dataset for training and testing
    '''
    def __init__(self, audio_dir, test_fold=None, sr=8000, file_duration=4, chunk_duration=0.1, train_class_pct=0.4):
        '''
        Constructor
        '''
        self.audio_dir = audio_dir
        self.sr = sr

        # Dataset classes
        self.classes = [
            'air_conditioner',
            'car_horn',
            'children_playing',
            'dog_bark',
            'drilling',
            'engine_idling',
            'gun_shot',
            'jackhammer',
            'siren',
            'street_music'
        ]

        # Determine which fold will be used as test set; all others used for training
        if test_fold is None:
            self.test_fold = random.randint(1, 10)
        elif 1 <= test_fold and test_fold <= 10:
            self.test_fold = test_fold
        else:
            raise ValueError('test_fold must be in range [1, 10]')

        # Compile lists of all training and testing file paths
        train_set_class_counts = [0 for c in self.classes]
        train_files = []
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
                    train_set_class_counts[self.get_label(f_path)] += 1
                    train_files.append(f_path)

        if train_class_pct < 0 or train_class_pct > 1:
            raise ValueError('train_class_pct must be within the range [0, 1)')

        # (Likely) Overselect the class being trained for on each model because
        # dataset is very unbalanced
        self.train_files_by_class = []
        for c in range(len(self.classes)):
            # True files must account for train_class_pct of training set
            p_true = train_class_pct/train_set_class_counts[c]
            # False files must account for (1 - train_class_pct) of training set
            p_false = (1 - train_class_pct)/(len(train_files) - train_set_class_counts[c])

            p = [p_true if self.get_label(f) == c else p_false for f in train_files]

            # Select training set
            self.train_files_by_class.append(np.random.choice(train_files, size=len(train_files),
                replace=True, p=p))

        # Batch size details
        self.file_duration = file_duration
        self.chunks = int(file_duration/chunk_duration)
        self.chunk_len = int(chunk_duration*self.sr)
        self.total_samples = self.sr*file_duration

        # Iterators to keep track of where we are in the training and testing sets
        self.i_train = [0 for c in self.classes]
        self.i_test = 0

    def get_batch(self, type, size=10, train_class=None, use_fft=False):
        '''
        Get next batch of shape (batch, seq_len, seq), which is representative
        of (file, chunks, chunk_len)
        '''
        # Gross hack to actually increment the iterator for the set being used
        if type == 'train':
            if train_class is not None and train_class in [c for c in range(len(self.classes))]:
                file_set = self.train_files_by_class[train_class]
                iterator = self.i_train[train_class]

                # Increment iterator
                self.i_train[train_class] += size
                if self.i_train[train_class] + size >= len(self.train_files_by_class[train_class]):
                    self.i_train[train_class] = 0
            else:
                raise ValueError('train_class must be in the range [0, 9]')
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
            labels.append(self.get_label(file))

            # Load data
            Y, sr = librosa.core.load(file, sr=self.sr, duration=self.file_duration)

            if Y.shape[0] < self.total_samples:
                # Pad this array with zeros on the end
                Y = np.pad(Y, (0, int(self.total_samples-Y.shape[0])), 'constant')

            # Chunk-up data
            for chunk in range(self.chunks):
                batch[i][chunk] = Y[chunk*self.chunk_len:chunk*self.chunk_len+self.chunk_len]

                if use_fft:
                    # Use FFT of chunk
                    batch[i][chunk] = fft(batch[i][chunk]).real

        return torch.from_numpy(batch.astype(np.float32)), labels

    def get_label(self, file):
        '''
        Return label of a given file
        '''
        return int(os.path.split(file)[-1].split('-')[1])

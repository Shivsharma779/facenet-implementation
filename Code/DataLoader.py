import numpy as np
import random
import os
from skimage import io
import torch
from torch.utils.data import Dataset

class TripletDataset(Dataset):

    def __init__(self, root_dir, labelToImageMapping, num_triplets, transform=None, phase="train"):

        self.root_dir = root_dir
        self.labelToImageMapping = labelToImageMapping
        self.num_triplets = num_triplets
        self.transform = transform
        self.phase = phase
        self.num_data = 0
        for key in labelToImageMapping:
            self.num_data += len(labelToImageMapping[key])
        if self.phase == "test":
            self.training_triplets = self.generate_triplets_test(self.labelToImageMapping, self.num_triplets)
        else:
            self.training_triplets = self.generate_triplets(self.labelToImageMapping, self.num_triplets)
    
    
    def generate_triplets_test(self, labelToImageMapping, num_triplets):
        triplets = []
        print("Generating triplets for test", num_triplets)

        self.classes = np.array(list(labelToImageMapping.keys()))

        for i in range(num_triplets):
            pos_class = np.random.choice(self.classes)
            neg_class = np.random.choice(self.classes)
            while neg_class == pos_class:
                neg_class = np.random.choice(self.classes)

            ianc = np.random.randint(0, len(labelToImageMapping[pos_class]))
            ipos = np.random.randint(0, len(labelToImageMapping[pos_class]))
            while ianc == ipos:
                ipos = np.random.randint(0, len(labelToImageMapping[pos_class]))
            ineg = np.random.randint(0, len(labelToImageMapping[neg_class]))

            triplets.append([np.where(self.classes == pos_class)[0][0], np.where(self.classes == neg_class)[0][0]
                             , labelToImageMapping[pos_class][ianc]
                             , labelToImageMapping[pos_class][ipos],
                              labelToImageMapping[neg_class][ineg]])
      

        return triplets



    def generate_triplets(self, labelToImageMapping, num_triplets):
        triplets = []
        print("Generating triplets", num_triplets)
        
        self.classes = np.array(list(labelToImageMapping.keys()))
        minSampleCount = 20
        print("Minimum sample count:", minSampleCount * len(self.classes))


        print("Image instances in each class", len(labelToImageMapping[self.classes[0]]))

        for i in self.classes:
            for j in range(minSampleCount):
                pos_class = i
                neg_class = np.random.choice(self.classes)
                while neg_class == pos_class:
                    neg_class = np.random.choice(self.classes)
                ianc = np.random.randint(0, len(labelToImageMapping[pos_class]))
                ipos = np.random.randint(0, len(labelToImageMapping[pos_class]))
                while ianc == ipos:
                    ipos = np.random.randint(0, len(labelToImageMapping[pos_class]))
                ineg = np.random.randint(0, len(labelToImageMapping[neg_class]))

                triplets.append([np.where(self.classes == pos_class)[0][0], np.where(self.classes == neg_class)[0][0]
                              , labelToImageMapping[pos_class][ianc]
                              , labelToImageMapping[pos_class][ipos],
                                labelToImageMapping[neg_class][ineg]])

        

        num_triplets -= len(triplets)
        random.shuffle(triplets)

        for i in range(num_triplets):
            pos_class = np.random.choice(self.classes)
            neg_class = np.random.choice(self.classes)
            while neg_class == pos_class:
              neg_class = np.random.choice(self.classes)

            ianc = np.random.randint(0, len(labelToImageMapping[pos_class]))
            ipos = np.random.randint(0, len(labelToImageMapping[pos_class]))
            while ianc == ipos:
                ipos = np.random.randint(0, len(labelToImageMapping[pos_class]))
            ineg = np.random.randint(0, len(labelToImageMapping[neg_class]))

            triplets.append([np.where(self.classes == pos_class)[0][0], np.where(self.classes == neg_class)[0][0]
                             , labelToImageMapping[pos_class][ianc]
                             , labelToImageMapping[pos_class][ipos],
                              labelToImageMapping[neg_class][ineg]])

      
      

        return triplets


    def regenerateTriplets(self):
        if self.phase == "test":
            self.training_triplets = self.generate_triplets_test(self.labelToImageMapping, self.num_triplets)
        else:
            self.training_triplets = self.generate_triplets(self.labelToImageMapping, self.num_triplets)

    def __getitem__(self, idx):

        pos_class, neg_class, anc_id, pos_id, neg_id = self.training_triplets[idx]

        anc_img = os.path.join(self.root_dir, anc_id)
        pos_img = os.path.join(self.root_dir, pos_id)
        neg_img = os.path.join(self.root_dir, neg_id)

        anc_img = io.imread(anc_img)
        pos_img = io.imread(pos_img)
        neg_img = io.imread(neg_img)


        pos_class = torch.from_numpy(np.array([pos_class]).astype('long'))
        neg_class = torch.from_numpy(np.array([neg_class]).astype('long'))

        sample = {'anc_img': anc_img, 'pos_img': pos_img, 'neg_img': neg_img, 'pos_class': pos_class,
                'neg_class': neg_class}

      # sample['anc_img'] = Image.fromarray(sample['anc_img'], mode='L')
      # sample['pos_img'] = Image.fromarray(sample['pos_img'], mode='L')
      # sample['neg_img'] = Image.fromarray(sample['neg_img'], mode='L')

        if self.transform:
            sample['anc_img'] = self.transform(sample['anc_img'])
            sample['pos_img'] = self.transform(sample['pos_img'])
            sample['neg_img'] = self.transform(sample['neg_img'])

        sample['anc_img'].requires_grad_(True)
        sample['pos_img'].requires_grad_(True)
        sample['neg_img'].requires_grad_(True)

        return sample

    def __len__(self):
        return len(self.training_triplets)
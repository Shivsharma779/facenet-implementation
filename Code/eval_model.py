from skimage import io
import numpy as np
import torch
import os

def all_pairs_euclid_torch(A, B):
	#
	sqrA = torch.sum(torch.pow(A, 2), 1, keepdim=True).expand(A.shape[0], B.shape[0])
	sqrB = torch.sum(torch.pow(B, 2), 1, keepdim=True).expand(B.shape[0], A.shape[0]).t()
	#
	return torch.sqrt(sqrA - 2*torch.mm(A, B.t()) + sqrB)


def testValues(model, embedding_size, transform, trainSetMapping, trainSet, loader):

    model.eval()

    model.cuda()

    no_samples_per_class = 5

    matchedPoints = 0

    w = 224
    h = 224
    c = 3


    randomClassSamples = np.array([np.random.choice(trainSetMapping[i], no_samples_per_class) for i in trainSet.classes])
    Xsample = torch.zeros(len(trainSet.classes), no_samples_per_class, c, w, h)

    for idxi, sample in enumerate(randomClassSamples):
        for idxj, img in enumerate(sample):
            Xsample[idxi][idxj] = transform(io.imread(os.path.join(trainSet.root_dir, img)))


    print(Xsample.shape)

    with torch.no_grad():
    #     tempEncodings = torch.zeros(no_samples_per_class, embedding_size)
        sampleEmbeddings = torch.zeros(len(trainSet.classes), embedding_size).cuda()

        for j in range(len(trainSet.classes)):
            sampleEmbeddings[j] = model(Xsample[j].cuda()).mean(dim = 0)

        totalPoints =  0
        for batch in loader:
            embedPred = model(batch["anc_img"].cuda())
            y_true = batch["pos_class"].flatten()
            distances = all_pairs_euclid_torch(embedPred, sampleEmbeddings)
            y_pred = torch.argmin(distances, dim=1).cpu()
            matchedPoints += torch.sum(y_pred == y_true)
            totalPoints += y_true.shape[0]



    model.train()

    model.cuda()

    return (matchedPoints / totalPoints)


    
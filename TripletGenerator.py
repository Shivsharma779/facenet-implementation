import torch
import torch.nn as nn

def generateHardNormalTriplets(network, anchor, positive, negative, margin,_type="train"):
    network.eval()

    with torch.no_grad():

        lossFn = nn.TripletMarginLoss(margin=margin, p=2, reduction='none')
        anchor_m, positive_m, negative_m = network(anchor), network(positive), network(negative)
        loss = lossFn(anchor_m, positive_m, negative_m)

        lossIndices = [i for i in range(anchor.shape[0])]

        if _type == "train":
            lossIndices = torch.topk(loss, 16)[1]

    anchor, positive, negative = anchor[lossIndices], positive[lossIndices], negative[lossIndices]

    network.train()
    return anchor, positive, negative
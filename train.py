import json
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from time import time
from eval_model import testValues
from DataLoader import TripletDataset
from DatasetMapping import getMappingVGG, getMappingLFW, splitIntoTrainValid
from model import ResnetFaceNet
from TripletGenerator import generateHardNormalTriplets

import matplotlib.image as mpimg
import os


def tripletLossTrain(loader, model, optimizer, criterion, margin):
    training_loss = 0
    
    for idx, batch in enumerate(loader):
        optimizer.zero_grad()
        batch["anc_img"], batch["pos_img"], batch["neg_img"] = batch["anc_img"].cuda(), batch["pos_img"].cuda(), batch["neg_img"].cuda()
        anchor, positive, negative = generateHardNormalTriplets(model, batch["anc_img"], batch["pos_img"], batch["neg_img"], margin,_type="train")
#         if anchor.shape[0] < 2:
#             continue
        
        anchor, positive, negative = model(anchor), model(positive), model(negative)
        loss = criterion(anchor, positive, negative)
        # print(loss)
        loss.backward()
        optimizer.step()
        
        training_loss += loss.item()
    return training_loss / len(loader)


def tripletLossValid(loader, model, criterion, margin):
    valid_loss = 0
    model.eval()
    
    with torch.no_grad():
        for idx, batch in enumerate(loader):
            batch["anc_img"], batch["pos_img"], batch["neg_img"] = batch["anc_img"].cuda(), batch["pos_img"].cuda(), batch["neg_img"].cuda()
            anchor, positive, negative = generateHardNormalTriplets(model, batch["anc_img"], batch["pos_img"], batch["neg_img"], margin,_type="valid")
            if anchor.shape[0] == 0:
                continue

            anchor, positive, negative = model(anchor), model(positive), model(negative)
            loss = criterion(anchor, positive, negative)
            
            valid_loss += loss.item()
        
    
    model.train()
    
    return valid_loss / len(loader)


def main():
    train_on_gpu = torch.cuda.is_available()
    with open("config.json") as f:
        config = json.load(f)
    batch_size = config["batch_size"]
    embedding_size = config["embedding_size"]
    epochs = config["epochs"]
    rootPath = config["rootPath"]
    rootPathLFW = config["rootPathLFW"]
    splitRatio = config["splitRatio"]
    margin = config["margin"]

    labelToImageMapping = getMappingVGG(rootPath)
    labelToImageMappingLFW = getMappingLFW(rootPathLFW)
    
    
    trainSetMapping, validSetMapping = splitIntoTrainValid(labelToImageMapping, splitRatio)
    

    transform = transforms.Compose([transforms.ToTensor(),
                                    #transforms.Resize(256),
                                    #transforms.CenterCrop(224),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                                ])

    trainSet = TripletDataset(rootPath, trainSetMapping, 10000, transform)
    validSet = TripletDataset(rootPath, validSetMapping, 4000, transform, phase="test")
    testSet = TripletDataset(rootPathLFW, labelToImageMappingLFW, 2000, transform, phase="test")

    train_loader = torch.utils.data.DataLoader(trainSet, batch_size=batch_size, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(validSet, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(testSet, batch_size=batch_size, shuffle=True)

    model = ResnetFaceNet(embedding_size, pretrained=True)
    #model.load_state_dict(checkpoint['model_state_dict'])

    if train_on_gpu:
        model.cuda()

    optimizer = torch.optim.Adam(model.parameters(),lr = 0.0003)
    #optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    criterion = nn.TripletMarginLoss(margin=margin, reduction="mean")
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')

    train(model, trainSetMapping, labelToImageMappingLFW, trainSet, testSet, train_loader , valid_loader, test_loader, optimizer, criterion, margin,epochs)


def train(model, trainSetMapping, labelToImageMappingLFW, trainSet, testSet, train_loader, valid_loader, test_loader, optimizer, criterion, margin,epochs):
    validationAccuracyList = []
    testAccuracyList = []

    lossListTrain = []
    lossListLFW = []

    epochs = 100

    training_loss = 0
    lfw_loss = 0
    maxAccuracy = 0

    for i in range(epochs):
        
        start = time()

        training_loss = tripletLossTrain(train_loader, model, optimizer, criterion, margin)
        lfw_loss = tripletLossValid(test_loader, model, criterion, margin)
        
        
        lossListTrain.append(training_loss)
        lossListLFW.append(lfw_loss)
    #     scheduler.step(lfw_loss)
        

        print("Epoch", i, "Training Loss:", training_loss)
        print("Epoch", i, "LFW Loss:", lfw_loss)
        
        validAccuracy = testValues(model, trainSetMapping, trainSet,valid_loader).item()
        testAccuracy = testValues(model, labelToImageMappingLFW, testSet,test_loader).item()
        
        validationAccuracyList.append(validAccuracy)
        testAccuracyList.append(testAccuracy)
        
        print("Validation accuracy:", validAccuracy)
        print("Test accuracy:", testAccuracy)
        
        if validAccuracy > maxAccuracy:
            maxAccuracy = validAccuracy
            
            torch.save({
                        'epoch': i,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict()
                        }, "checkpoint_gnet.pth")
            

        print("Total time:", (time() - start))
        
        if i % 15 == 0:
            with open("validationAccuracyList.picke", "wb") as f:
                pickle.dump(validationAccuracyList, f)
                
            with open("testAccuracyList.picke", "wb") as f:
                pickle.dump(testAccuracyList, f)
                
            with open("lossListTrain.picke", "wb") as f:
                pickle.dump(lossListTrain, f)
                
            with open("lossListLFW.picke", "wb") as f:
                pickle.dump(lossListLFW, f)
        
        trainSet.regenerateTriplets()

    with open("validationAccuracyList.picke", "wb") as f:
        pickle.dump(validationAccuracyList, f)
        
    with open("testAccuracyList.picke", "wb") as f:
        pickle.dump(testAccuracyList, f)
        
    with open("lossListTrain.picke", "wb") as f:
        pickle.dump(lossListTrain, f)
        
    with open("lossListLFW.picke", "wb") as f:
        pickle.dump(lossListLFW, f)


if __name__ == "__main__":
    main()
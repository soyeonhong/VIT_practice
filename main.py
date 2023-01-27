import torch.optim as optim
import torch.nn as nn
import torch
from data import dataload
from model import ViT

train, test=dataload(32)

device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

vit=ViT(
    image_size = 256,
    patch_size = 32,
    num_classes = 10,
    dim = 1024,
    depth = 6,
    heads = 16,
    mlp_dim = 2048,
    dropout = 0.1,
    emb_dropout = 0.1   ).to(device)

loss_fn=nn.CrossEntropyLoss()
optimizer=optim.Adam(vit.parameters(),lr=0.001)
epochs=10

def train_loop(dataloader, model, loss_fn, optimizer):
    size=len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader):
        
        # Prediction & loss
        X=X.to(device)
        y=y.to(device)

        pred=model(X)
        loss=loss_fn(pred,y)

        # Backpropagation
        optimizer.zero_grad() # 매개변수 변화도 재설정
        loss.backward() # 예측 손실 역전파
        optimizer.step # 역전파 단계에서 수집된 변화도로 매개변수 조정

        if batch % 100==0:
            loss, current=loss.item(), batch*len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

def test_loop(dataloader, model, loss_fn):
    size=len(dataloader.dataset)
    num_batches=len(dataloader)
    test_loss, correct=0,0

    with torch.no_grad():
        for X, y in dataloader:
            X, y=X.to(device), y.to(device)
            pred=model(X)
            test_loss=loss_fn(pred, y).item()
            correct+=(pred.argmax(1)==y).type(torch.float).sum().item()

    test_loss/=num_batches
    correct/=size

    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train_loop(train, vit, loss_fn, optimizer)
    test_loop(test, vit, loss_fn)

print("Done!")





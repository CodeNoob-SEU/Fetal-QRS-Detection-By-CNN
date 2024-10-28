import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from dataset import create_dataset
from net import *

learning_rate = 0.001
batch_size = 2048
num_epoch = 100
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model_path = './model.ckpt'

train_set, valid_set, test_set = create_dataset()
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
valid_loader = DataLoader(valid_set, batch_size=batch_size, shuffle=True)

ll = train_set[0][0]
print(ll.dtype)
print(ll.shape)
net = Net().to(device)
criterion = nn.CrossEntropyLoss().to(device)
optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)


def train():
    stale = 0
    best_acc = 0
    for epoch in range(num_epoch):
        net.train()
        train_loss = []
        train_accs = []
        for imgs, labels in tqdm(train_loader):
            # print(imgs.shape)
            imgs = imgs.unsqueeze(1)
            imgs, labels = imgs.to(device), labels.to(device)
            pred = net(imgs)
            # print(pred.shape)
            # print(labels.shape)
            loss = criterion(pred, labels)
            # print(pred.shape,labels.shape)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss.append(loss.item())
            acc = (pred.argmax(dim=-1) == labels.to(device)).float().mean()
            train_accs.append(acc)
        train_loss = sum(train_loss) / len(train_loss)
        train_acc = sum(train_accs) / len(train_accs)
        print(f"[ Train | {epoch + 1:03d}/{num_epoch:03d} ] loss = {train_loss:.5f}, acc = {train_acc:.5f}")

        net.eval()
        valid_loss = []
        valid_accs = []
        for imgs, labels in tqdm(valid_loader):
            imgs = imgs.unsqueeze(1)
            imgs, labels = imgs.to(device), labels.to(device)

            with torch.no_grad():
                pred = net(imgs)
            loss = criterion(pred, labels)
            acc = (pred.argmax(dim=-1) == labels).float().mean()

            # Record the loss and accuracy.
            valid_loss.append(loss.item())
            valid_accs.append(acc)
            # break

        # The average loss and accuracy for entire validation set is the average of the recorded values.
        valid_loss = sum(valid_loss) / len(valid_loss)
        valid_acc = sum(valid_accs) / len(valid_accs)

        # Print the information.
        print(f"[ Valid | {epoch + 1:03d}/{num_epoch:03d} ] loss = {valid_loss:.5f}, acc = {valid_acc:.5f}")
        if valid_acc > best_acc:
            print(f"[ Valid | {epoch + 1:03d}/{num_epoch:03d} ] loss = {valid_loss:.5f}, acc = {valid_acc:.5f} -> best")
        else:
            print(f"[ Valid | {epoch + 1:03d}/{num_epoch:03d} ] loss = {valid_loss:.5f}, acc = {valid_acc:.5f}")

        # save models
        if valid_acc > best_acc:
            print(f"Best model found at epoch {epoch}, saving model")
            torch.save(net.state_dict(), "best.ckpt")  # only save best to prevent output memory exceed error
            best_acc = valid_acc
            stale = 0


train()


# train(net, train_loader,valid_loader, criterion, optimizer)

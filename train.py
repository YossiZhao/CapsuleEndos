
import torch
import torchvision
from utilis.DataLoad import customDataset
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torch.optim as optim
from torch import nn
#
from utilis.config import parse_args
from models.Resnet import resnet34
# from utilis.DataLoad import Traindataloader
# from models.VGG16 import VGG16



# Get cpu or gpu device for training.
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using {} device".format(device))




dataset = customDataset(csv_file='./data/label_CapsuleEndos.csv', data_dir='./data/CapsuleEnds', \
                        transform=transform)

train_set, test_set = torch.utils.data.random_split(dataset, [40000, 7388])
'''
    This part is used to save or load model.
'''
def save_checkpoint(model, save_path='./runs/train/'):
    print("=> Saving checkpoint")
    torch.save(model.state_dict(), save_path+'model_weights.pth')

def load_checkpoint(model, load_path):
    print("=> Loading checkpoint")
    model.load_state_dict(torch.load(load_path+'model_weights.pth'))

# check performance of model.
def check_accuracy(model, loader):
    num_correct = 0
    num_samples = 0
    model.eval()

    with torch.no_grad():
        for images, labels in loader:
            images = images.to('cuda')
            labels = labels.to('cuda')
            results = model(images)
            _, predictions = results.max(1)
            num_correct += (predictions == labels).sum()
            num_samples += predictions.size(0)

        print(f'Got {num_correct}/{num_samples} with accuracy \
            {float(num_correct)/float(num_samples)*100}%')






def train(args):

    # Load Data
    # transform = transforms.Compose([
    #                                    transforms.Resize((args.input_size, args.input_size)),
    #                                    transforms.ToTensor(),
    # ])
    # dataset = customDataset(csv_file='./data/label.csv', data_dir='./data/Stomach', \
    #                         transform=transform)

    # train_set, test_set = torch.utils.data.random_split(dataset, [22000, 1067])
    train_loader = DataLoader(dataset=train_set, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_set, batch_size=args.batch_size, shuffle=True)
    # dataiter = iter(train_loader)


    # Initialization: Model, Loss Function, Optimizer, Visualization

    model = resnet34(args.num_classes).to(device)
    criterion = nn.CrossEntropyLoss()     # initialize loss function
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)    # initialize optimizer


    # Set checkpoint
    checkpoint = {'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()}

    # Train network
    for epoch in range(args.epochs):
        running_loss = 0.0
        for batch_index, (images, targets) in enumerate(train_loader, 0):
            # images, labels = dataiter.next()
            print('The shape of images is :', images.shape)
            images = images.to(device)
            targets = targets.to(device)
            # forward + backward + optimize
            outputs = model(images)
            loss = criterion(outputs, targets)
            # zero the parameter gradients
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        check_accuracy(model, test_loader)
            # print statistics
            # running_loss += loss.item()
            # if i % 2000 == 1999:  # print every 2000 mini-batches
            #     print('[%d, %5d] loss: %.3f' %  (epoch + 1, i + 1, running_loss / 2000))
            #
            #     running_loss = 0.0
        # model.eval()
        # test_images =
    # save_checkpoint(model)

if __name__ == '__main__':
    args = parse_args()
    train(args)
    print('Training completed')






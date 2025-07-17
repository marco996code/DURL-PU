import  argparse
import torch
from  torch import  nn
import numpy as np
from utils import *
from collections import defaultdict
import torchvision.models as models
# from resnet import  ResNet18
def inference(loader, model, device):
    feature_vector = []
    labels_vector = []
    model.eval()
    for step, (x, y) in enumerate(loader):
        x = x.to(device)

        # get encoding
        with torch.no_grad():
            h = model(x)

        h = h.squeeze()
        h = h.detach()

        feature_vector.extend(h.cpu().detach().numpy())
        labels_vector.extend(y.numpy())

        if step % 5 == 0:
            print(f"Step [{step}/{len(loader)}]\t Computing features...")

    feature_vector = np.array(feature_vector)
    labels_vector = np.array(labels_vector)
    print("Features shape {}".format(feature_vector.shape))
    return feature_vector, labels_vector


def get_features(model, train_loader, test_loader, device):
    train_X, train_y = inference(train_loader, model, device)
    test_X, test_y = inference(test_loader, model, device)
    return train_X, train_y, test_X, test_y


def create_data_loaders_from_arrays(X_train, y_train, X_test, y_test, batch_size):
    train = torch.utils.data.TensorDataset(
        torch.from_numpy(X_train), torch.from_numpy(y_train)
    )
    train_loader = torch.utils.data.DataLoader(
        train, batch_size=batch_size, shuffle=False
    )

    test = torch.utils.data.TensorDataset(
        torch.from_numpy(X_test), torch.from_numpy(y_test)
    )
    test_loader = torch.utils.data.DataLoader(
        test, batch_size=batch_size, shuffle=False
    )
    return train_loader, test_loader

def get_encoder_network(model, encoder_network, num_classes=10, projection_size=2048, projection_hidden_size=4096):
    if encoder_network == "resnet18":
        f= []
        full_size = False
        pretrained_model_name = 'resnet18'
        # resnet = eval(f'models.{pretrained_model_name}')(pretrained=False)
        resnet = models.resnet18(pretrained=False)
        #resnet = ResNet18(num_classes=num_classes)
        for name, module in resnet.named_children():

            if name == "conv1" and not full_size:  # add not full_size
                module = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)

            if full_size:
                print(name)
                if name != "fc":
                    f.append(module)
            else:
                if not isinstance(module, nn.Linear) and not isinstance(
                        module, nn.MaxPool2d
                ):
                    f.append(module)
        # encoder
        f = nn.Sequential(*f)
        feat_dim = 512
        print("feat dim:", feat_dim)

    # resnet.f = f
    # if model in [Symmetric, SimSiam, BYOL, SymmetricNoSG, SimSiamNoSG, BYOLNoSG, SimCLR]:
    #     resnet.fc = MLP(resnet.feature_dim, projection_size, projection_hidden_size)
    # if model == MoCoV2:
    #     resnet.fc = MLP(resnet.feature_dim, num_classes, resnet.feature_dim)

    return f

class FeatureExtractor(nn.Module):
    def __init__(self, args):
        super(FeatureExtractor, self).__init__()
        self.f = get_encoder_network(args.model, args.encoder_network)
    
    def forward(self, x):
        x = self.f(x)
        x = torch.flatten(x, start_dim=1)
        return x
    
def test_result(test_loader, logreg, device, model_path):
    # Test fine-tuned model
    print("### Calculating final testing performance ###")
    logreg.eval()
    metrics = defaultdict(list)
    for step, (h, y) in enumerate(test_loader):
        h = h.to(device)
        y = y.to(device)

        outputs = logreg(h)

        # calculate accuracy and save metrics
        accuracy = (outputs.argmax(1) == y).sum().item() / y.size(0)
        metrics["Accuracy/test"].append(accuracy)

    print(f"Final test performance: " + model_path)
    for k, v in metrics.items():
        print(f"{k}: {np.array(v).mean():.4f}")
    return np.array(metrics["Accuracy/test"]).mean()


if __name__ == "__main__":
     parser = argparse.ArgumentParser()
     parser.add_argument("--dataset", default="ucm", type=str)
     parser.add_argument("--model_path", default='da', type=str, help="Path to pre-trained model (e.g. model-10.pt)")
     parser.add_argument('--model', default='simsiam', type=str, help='name of the network')
     parser.add_argument("--image_size", default=64, type=int, help="Image size")
     parser.add_argument("--learning_rate", default=5e-3, type=float, help="Initial learning rate.")
     parser.add_argument("--batch_size", default=512, type=int, help="Batch size for training.")
     parser.add_argument("--num_epochs", default=500, type=int, help="Number of epochs to train for.")
     parser.add_argument("--encoder_network", default="resnet18", type=str, help="Encoder network architecture.")
     parser.add_argument("--num_workers", default=8, type=int, help="Number of data workers (caution with nodes!)")
     parser.add_argument("--fc", default="identity", help="options: identity, remove")
     parser.add_argument("--nor", type=int, default=1, help="set 1 reprenset use normalization in test")
     args = parser.parse_args()

     args.model = 'byol'
     device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

     # get data loaders
     train_loader, test_loader = get_data_loaders(args.nor, args.dataset, args.image_size, args.batch_size, args.num_workers)

     resnet = FeatureExtractor(args)

     state_dict = torch.load(args.model_path)
     state_dict = state_dict['model']

     resnet.load_state_dict(state_dict, strict=False)
     resnet = resnet.to(device)
     num_features = 512
     if args.fc == "remove":
         resnet = nn.Sequential(*list(resnet.children())[:-1])  # throw away fc layer
     else:
         resnet.fc = nn.Identity()

     n_classes = 10
     if args.dataset == "cifar10":
         n_classes = 100
     elif args.dataset == "aid":
         n_classes = 30
     elif args.dataset == "ucm":
         n_classes = 21
     elif args.dataset == "fairm":
         n_classes = 30
     elif args.dataset == "nwpu":
         n_classes = 45
     # fine-tune model
     logreg = nn.Sequential(nn.Linear(num_features, n_classes))
     logreg = logreg.to(device)

    # loss / optimizer
     criterion = nn.CrossEntropyLoss()
     optimizer = torch.optim.Adam(params=logreg.parameters(), lr=args.learning_rate)

     # compute features (only needs to be done once, since it does not backprop during fine-tuning)
     print("Creating features from pre-trained model")
     (train_X, train_y, test_X, test_y) = get_features(
        resnet, train_loader, test_loader, device
     )

     train_loader, test_loader = create_data_loaders_from_arrays(
        train_X, train_y, test_X, test_y, 2048
     )

     # Train fine-tuned model
     logreg.train()
     for epoch in range(args.num_epochs):
         metrics = defaultdict(list)
         for step, (h, y) in enumerate(train_loader):
             h = h.to(device)
             y = y.to(device)

             outputs = logreg(h)

             loss = criterion(outputs, y)
             optimizer.zero_grad()
             loss.backward()
             optimizer.step()

             # calculate accuracy and save metrics
             accuracy = (outputs.argmax(1) == y).sum().item() / y.size(0)
             metrics["Loss/train"].append(loss.item())
             metrics["Accuracy/train"].append(accuracy)

         print(f"Epoch [{epoch}/{args.num_epochs}]: " + "\t".join(
             [f"{k}: {np.array(v).mean()}" for k, v in metrics.items()]))

         if epoch % 100 == 0:
             print("======epoch {}======".format(epoch))
             test_result(test_loader, logreg, device, args.model_path)
     test_result(test_loader, logreg, device, args.model_path)


import torch
import torchvision

class Resnet50(torch.nn.Module):
    """
    vehicle multilabel-classifier
    """

    def __init__(self, num_cls, input_size):
        """
        :param is_freeze:
        """
        torch.nn.Module.__init__(self)

        # output channels
        self._num_cls = num_cls

        # input image size
        self.input_size = input_size


        # delete origin FC and add custom FC
        self.features = torchvision.models.resnet50(pretrained=True)  # True
        del self.features.fc
        # print('feature extractor:\n', self.features)

        self.features = torch.nn.Sequential(
            *list(self.features.children()))

        self.fc = torch.nn.Linear(2048 ** 2, num_cls)  # output channels
        # print('=> fc layer:\n', self.fc)


    def forward(self, X):
        """
        :param X:
        :return:
        """
        N = X.size()[0]

        # assert X.size() == (N, 3, self.input_size, self.input_size)

        X = self.features(X)  # extract features

        # print('X.size: ', X.size())
        # assert X.size() == (N, 512, 1, 1)

        X = X.view(N, 2048, 1)
        X = torch.bmm(X, torch.transpose(X, 1, 2))  # 特征自卷集，扩充特征的维度

        # assert X.size() == (N, 512, 512)

        X = X.view(N, 2048 ** 2)
        X = torch.sqrt(X + 1e-5)
        X = torch.nn.functional.normalize(X)
        X = self.fc(X)
        return X

def get_predict(output):
    """
    get prediction from output
    """
    # get each label's prediction from output
    output = output.cpu()  # fetch data from gpu
    pred = output.max(1, keepdim=True)[1]

    return pred
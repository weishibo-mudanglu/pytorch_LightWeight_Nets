import torch
import torchvision

class Backbone(torch.nn.Module):
    """
    vehicle multilabel-classifier
    """

    def __init__(self,input_size,PreModel):
        """
        :param is_freeze:
        """
        torch.nn.Module.__init__(self)

        model = torch.jit.load(PreModel)
        self.features=model


    def forward(self, X):
        """
        :param X:
        :return:
        """
        N = X.size()[0]
        X = self.features(X) 

        # print('X.size: ', X.size())
        # assert X.size() == (N, 512, 1, 1)
        # X = X.view(N, 1024, 1)
        X = X.view(N, 1024)
        # X = torch.bmm(X, torch.transpose(X, 1, 2))  # 特征自卷集，扩充特征的维度
        return X

class Simamese(torch.nn.Module):
    def __init__(self,input_size,PreModel) -> None:
        torch.nn.Module.__init__(self)
        self.backbon=Backbone(input_size=input_size,PreModel=PreModel)


    def forward(self,X1,X2):
        
        X1 = self.backbon(X1)
        X2 = self.backbon(X2)
        # X = torch.concat(X1,X2,dim=3)
        X = torch.nn.functional.cosine_similarity(X1,X2,dim=1)
        return X
    def get_predict(X,thresh):
        if X < thresh:
            return 0
        else:
            return 1

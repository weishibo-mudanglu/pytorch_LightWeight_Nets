import torch
import torchvision
import math
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
        # X = X.view(N,4,256)
        # X = torch.bmm(X, torch.transpose(X, 1, 2))  # 特征自卷集，扩充特征的维度
        return X

class selfAttention(torch.nn.Module) :
    def __init__(self, num_attention_heads, input_size, hidden_size):
        super(selfAttention, self).__init__()
        if hidden_size % num_attention_heads != 0 :
            raise ValueError(
                "the hidden size %d is not a multiple of the number of attention heads"
                "%d" % (hidden_size, num_attention_heads)
            )

        self.num_attention_heads = num_attention_heads
        self.attention_head_size = int(hidden_size / num_attention_heads)
        self.all_head_size = hidden_size

        self.key_layer = torch.nn.Linear(input_size, hidden_size,dtype=torch.float16)
        self.query_layer = torch.nn.Linear(input_size, hidden_size,dtype=torch.float16)
        self.value_layer = torch.nn.Linear(input_size, hidden_size,dtype=torch.float16)

    def trans_to_multiple_heads(self, x):
        new_size = x.size()[ : -1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(new_size)
        return x.permute(0, 2, 1, 3)

    def forward(self, x):
        key = self.key_layer(x)
        query = self.query_layer(x)
        value = self.value_layer(x)

        key_heads = self.trans_to_multiple_heads(key)
        query_heads = self.trans_to_multiple_heads(query)
        value_heads = self.trans_to_multiple_heads(value)

        attention_scores = torch.matmul(query_heads, key_heads.permute(0, 1, 3, 2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        attention_probs = torch.nn.functional.softmax(attention_scores, dim = -1)

        context = torch.matmul(attention_probs, value_heads)
        context = context.permute(0, 2, 1, 3).contiguous()
        new_size = context.size()[ : -2] + (self.all_head_size , )
        context = context.view(new_size[0],-1)
        return context


class Simamese(torch.nn.Module):
    def __init__(self,input_size,PreModel) -> None:
        torch.nn.Module.__init__(self)
        self.backbon=Backbone(input_size=input_size,PreModel=PreModel)
        # self.selfattention=selfAttention(num_attention_heads=8,input_size=256,hidden_size=64)

    def forward(self,X1,X2):
        
        X1 = self.backbon(X1)
        X2 = self.backbon(X2)
        # X1 = self.selfattention(X1)
        # X2 = self.selfattention(X2)
        # X = torch.concat(X1,X2,dim=3)
        X = torch.nn.functional.cosine_similarity(X1,X2,dim=1)
        # return X1,X2
        return X
    def get_predict(X1,X2):

        X = torch.nn.functional.cosine_similarity(X1,X2,dim=1)
        if X <=0:
            return 1
        else:
            return 0

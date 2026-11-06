from torch import nn
from acti.tokenization import LABELS
from acti.models.crf import BIOESCRF


class EntityHead(nn.Module):
    def __init__(self,hidden_size):
        super().__init__();self.linear=nn.Linear(hidden_size,len(LABELS));self.crf=BIOESCRF()

    def forward(self,hidden):return self.linear(hidden)

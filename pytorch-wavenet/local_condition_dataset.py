import torch

class LocalConditionedDataset(torch.utils.data.Dataset):
  def __init__(self, data, local_condition, train, target_length):
    self.data = data
    self.local_condition = local_condition

    self.train = train
    self.target_length = target_length

    self.data.train = self.train
    self.data.target_length = target_length

    if self.local_condition is not None:
      self.local_condition.train = self.train
      self.local_condition.target_length = self.target_length

  def __getitem__(self, index):
    x = self.data[index]
    if self.local_condition is None:
      return x

    local_condition = self.local_condition[index][0]
    return x, local_condition

  def __len__(self):
    return len(self.data)
  
  def set_train(self, train):
    self.train = train
    self.data.train = self.train
    if self.local_condition is not None:
      self.local_condition.train = self.train
  
  def get_train(self):
    return self.train

  def get_target_length(self):
    return self.target_length

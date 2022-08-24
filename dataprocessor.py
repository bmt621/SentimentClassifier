from torch import LongTensor,tensor
import numpy as np

class DataSet():
  def __init__(self,dataframe,tokenizer):
    
    data_x = list(dataframe['words'].values)
    self.labels = LongTensor(dataframe['label'].values)
    self.encodings = tokenizer(data_x,max_length = 500,truncation=True, padding=True)
    del data_x
  def __getitem__(self,idx):
    
    item = {key:tensor(val[idx]) for key, val in self.encodings.items()}
    item['labels'] = self.labels[idx]

    return (item)

  def __len__(self):
    return len(self.labels)


def apply_labels(df):
  new_df = df.copy()
  new_df['label'] = np.nan

  
  get_index = df[df['sentiment']=='Positive'].index
  new_df.loc[get_index,'label'] = 2

  get_index = df[df['sentiment']=='Negative'].index
  new_df.loc[get_index,'label'] = 1

  get_index = df[df['sentiment'] == "Neutral"].index
  new_df.loc[get_index,'label'] = 0

  new_df = new_df.drop("sentiment",1)
  return new_df


'''
from dataprocessor import *
from distilbert import DistilBertModelForClassification
import pandas as pd


#                          INFERENCE

labels = 3
words = ['Neutral','Positive','Negetive']
model_path = 'path/to/model
tokenizer_path = 'path/to/tokenizer'

model = DistilBertModelForClassification(num_labels=labels,label_words=words,model_path=model_path,tokenizer_path=tokenizer)

example_text = "this model is good"

model.infer(example_text)  # output: Positive

#                          TRAINING

path_to_csv = "path/to/csv"
df = pd.read_csv(path_to_csv)
df = df[['words','sentiment']]

new_df = apply_labels(df)

train_dataset = DataSet(new_df)

model.set_trainer(train_dataset,batch_size=32,epoch=30)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

model.set_device(device)
model.train()


'''
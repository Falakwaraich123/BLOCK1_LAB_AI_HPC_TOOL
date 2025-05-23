import json
from pathlib import Path
import torch
from torch.utils.data import DataLoader
import time
from tqdm import tqdm
import warnings
import datetime
import time
import os
from transformers import BertForQuestionAnswering
from torch.optim import AdamW
import pytorch_lightning as pl
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor, Timer
from pytorch_lightning.callbacks import Callback
from lightning.pytorch.strategies import DDPStrategy, FSDPStrategy, DeepSpeedStrategy
from lightning.pytorch.profilers import SimpleProfiler, AdvancedProfiler
from pytorch_lightning.loggers import TensorBoardLogger
import argparse



pathTrainData = "../data/train-v2.0.json"
pathTestData =  "../data/dev-v2.0.json"

def load_data(file_path):
    with open(file_path, "r") as f:
        data = json.load(f)

    contexts = []
    questions = []
    answers = []

    for group in data['data']:
        for passage in group['paragraphs']:
            context = passage['context']
            for qa in passage['qas']:
                question = qa['question']
                for answer in qa['answers']:
                    contexts.append(context)
                    questions.append(question)
                    answers.append(answer)
    
    return contexts, questions, answers

train_contexts, train_questions, train_answers = load_data(pathTrainData)
test_contexts, test_questions, test_answers = load_data(pathTestData)


def adjust_answer_indices(answers, contexts):

    for answer, context in zip(answers, contexts):
        real_answer = answer['text']
        start_idx = answer['answer_start']
        end_idx = start_idx + len(real_answer)


        if context[start_idx:end_idx] == real_answer:
            answer['answer_end'] = end_idx

        elif context[start_idx-1:end_idx-1] == real_answer:
            answer['answer_start'] = start_idx - 1
            answer['answer_end'] = end_idx - 1

        elif context[start_idx-2:end_idx-2] == real_answer:
            answer['answer_start'] = start_idx - 2
            answer['answer_end'] = end_idx - 2

adjust_answer_indices(train_answers, train_contexts)
adjust_answer_indices(test_answers, test_contexts)


from transformers import AutoTokenizer,AdamW,BertForQuestionAnswering
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
warnings.filterwarnings("ignore")

train_encodings = tokenizer(
    train_contexts, 
    train_questions, 
    truncation=True, 
    padding=True,
)

test_encodings = tokenizer(
    test_contexts, 
    test_questions, 
    truncation=True, 
    padding=True,
)


def add_token_positions(encodings, answers):
  start_positions = []
  end_positions = []

  count = 0

  for i in range(len(answers)):
    start_positions.append(encodings.char_to_token(i, answers[i]['answer_start']))
    end_positions.append(encodings.char_to_token(i, answers[i]['answer_end']))
    if start_positions[-1] is None:
      start_positions[-1] = tokenizer.model_max_length
    if end_positions[-1] is None:
      end_positions[-1] = encodings.char_to_token(i, answers[i]['answer_end'] - 1)
      if end_positions[-1] is None:
        count += 1
        end_positions[-1] = tokenizer.model_max_length

  encodings.update({'start_positions': start_positions, 'end_positions': end_positions})

add_token_positions(train_encodings, train_answers)
add_token_positions(test_encodings, test_answers)

class SquadDataset(torch.utils.data.Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __getitem__(self, idx):
        return {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}

    def __len__(self):
        return len(self.encodings.input_ids)

train_dataset = SquadDataset(train_encodings)
test_dataset = SquadDataset(test_encodings)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=True)

small_train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=8,
    sampler=torch.utils.data.SubsetRandomSampler(range(int(0.05 * len(train_dataset))))
)

small_test_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=8,
    sampler=torch.utils.data.SubsetRandomSampler(range(int(0.05 * len(test_dataset))))
)
torch.set_float32_matmul_precision('high')

class TimeCallback(Callback):
    def on_train_start(self, trainer, pl_module):
        self.start_time = time.time()
        print(f"\nTraining started at {time.strftime('%H:%M:%S')}")

    def on_train_end(self, trainer, pl_module):
        end_time = time.time()
        total_time = end_time - self.start_time
        minutes = int(total_time // 60)  # Convert total time to minutes
        seconds = total_time % 60  # Get remaining seconds
        print(f"\nTraining finished at {time.strftime('%H:%M:%S')}")
        print(f"Total training time: {minutes} minutes {seconds:.2f} seconds")


class MySlurmCallback(Callback):
    def on_train_start(self, trainer, pl_module):
        slurm_id = os.getenv('SLURM_JOB_ID')
        slurm_rank = os.getenv('SLURM_PROCID')
        device_id = torch.cuda.current_device()
        print(f"SLURM_JOB_ID: {slurm_id}, SLURM_PROCID: {slurm_rank}, CUDA Device ID: {device_id}")


class BertLightning(pl.LightningModule):
    def __init__(self, model_name='bert-base-uncased', learning_rate=5e-5):
        super().__init__()
        self.model = BertForQuestionAnswering.from_pretrained(model_name)
        self.learning_rate = learning_rate
        self.avg_train_loss = 0.0
        self.current_loss = 0.0
        self.val_loss = 0.0
        self.num_batches = 0

    def forward(self, input_ids, attention_mask, start_positions, end_positions):
        return self.model(input_ids, attention_mask=attention_mask, start_positions=start_positions, end_positions=end_positions)

    def training_step(self, batch, batch_idx):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        start_positions = batch['start_positions']
        end_positions = batch['end_positions']
        outputs = self(input_ids, attention_mask, start_positions, end_positions)
        loss = outputs[0]
        self.log('train_loss', loss)
        self.avg_train_loss += loss.item()
        self.current_loss = loss.item()
        self.num_batches += 1
        return loss

    def on_train_epoch_end(self):
        avg_loss = self.avg_train_loss / self.num_batches if self.num_batches > 0 else 0
        self.log('avg_train_loss', avg_loss)
        print(f"Epoch {self.current_epoch + 1}:")
        print(f"  - Current train loss (last batch): {self.current_loss:.4f}")
        print(f"  - Average train loss (last epoch): {avg_loss:.4f}")
        self.avg_train_loss = 0.0
        self.num_batches = 0

    def validation_step(self, batch, batch_idx):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        start_positions = batch['start_positions']
        end_positions = batch['end_positions']

        outputs = self(input_ids, attention_mask, start_positions, end_positions)
        loss = outputs[0]

        self.log('val_loss', loss)

    def configure_optimizers(self):
        return AdamW(self.model.parameters(), lr=self.learning_rate)

model = BertLightning()

tensorboard_logger = TensorBoardLogger("tb_logs", name="my_model")
parser = argparse.ArgumentParser(description="Choose a distributed strategy")
parser.add_argument('--strategy', type=str, default='ddp', choices=['ddp', 'fsdp', 'deepspeed'], help="Choose work distribution strategy: ddp, fsdp, deepspeed")
args = parser.parse_args()

if args.strategy == 'ddp':
    strategy = "ddp"
elif args.strategy == 'fsdp':
    strategy = "fsdp"
elif args.strategy == 'deepspeed':
    strategy = "deepspeed"

trainer = Trainer(
    max_epochs=2, 
    num_nodes=2,  
    accelerator="gpu",
    devices=2,  
    strategy=strategy, 
    profiler=SimpleProfiler(),
    logger=tensorboard_logger,
    callbacks=[
        EarlyStopping(monitor='train_loss'), 
        ModelCheckpoint(dirpath='checkpoints/', filename='{epoch}-{train_loss:.2f}'), 
        LearningRateMonitor(logging_interval='step'), 
        Timer(),
        MySlurmCallback(),
        TimeCallback()
    ]
)

trainer.fit(model, train_loader, test_loader)

print(f"\nWork distribution method used: {trainer.strategy.__class__.__name__}")

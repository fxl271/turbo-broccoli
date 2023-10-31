from transformers import TrainingArguments
from transformers import AutoTokenizer
from datasets import load_dataset
from transformers import DataCollatorWithPadding
from transformers import Trainer
from transformers import TrainerCallback
from transformers import AutoModelForSequenceClassification
from codecarbon import track_emissions
import time


@track_emissions(project_name="train_model")
def train_model(model_name, dataset_name, limit_size=True, output_dir="path/to/save/folder/", learning_rate=2e-5,per_device_train_batch_size=8,per_device_eval_batch_size=8, num_train_epochs=2):
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    
    
    training_args = TrainingArguments(
        output_dir,
        learning_rate,
        per_device_train_batch_size,
        per_device_eval_batch_size,
        num_train_epochs,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def tokenize_dataset(dataset):
        return tokenizer(dataset["text"])

    if (limit_size):
        dataset = load_dataset(dataset_name)['train'].train_test_split(train_size=400, test_size=100)
    else:
        dataset = load_dataset(dataset_name)

    dataset = dataset.map(tokenize_dataset, batched=True)

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    class My_Callback_Transformers(TrainerCallback):
        def on_train_begin(self, args, state, control, **kwargs):
            self.train_startTime = time.time()

            self.train_total = 0
            self.epoch_total = 0
            self.step_total = 0

        def on_epoch_begin(self, args, state, control, **kwargs):
            self.epoch_startTime = time.time()

        def on_step_begin(self, args, state, control, **kwargs):
            self.step_startTime = time.time()

        def on_step_end(self, args, state, control, **kwargs):
            self.step_endTime = time.time()
            self.step_time = self.step_endTime - self.step_startTime
            self.step_total = self.step_total + self.step_time

        def on_epoch_end(self, args, state, control, **kwargs):
            self.epoch_endTime = time.time()
            self.epoch_time = self.epoch_endTime - self.epoch_startTime
            self.epoch_total = self.epoch_total + self.epoch_time
            print(f"\tEPOCH TIME: {round(self.epoch_time,5)}s")

        def on_train_end(self, args, state, control, **kwargs):
            self.train_endTime = time.time()
            self.train_time = self.train_endTime - self.train_startTime
            self.train_total = self.train_total + self.train_time
            print(f"\t\tTRAIN TIME: {round(self.train_total,5)}s")
            print(f"\t\t\tTOTAL STEP: {round(self.step_total,5)}s")
            print(f"\t\t\tTOTAL EPOCH: {round(self.epoch_total,5)}s")

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        callbacks=[My_Callback_Transformers()]
    )
    
    trainer.train()
    
#train_model("distilbert-base-uncased", "rotton_tomatos")
    
#train_model("bigscience/bloomz-560m", "bigscience/xP3")
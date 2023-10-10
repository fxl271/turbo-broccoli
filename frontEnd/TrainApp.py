from transformers import TrainingArguments
from transformers import AutoTokenizer
from datasets import load_dataset
from transformers import DataCollatorWithPadding
from transformers import Trainer
from transformers import TrainerCallback
import time


def train_model(dataset_name, model_name, limit_size=True, output_dir="path/to/save/folder/", learning_rate=2e-5,per_device_train_batch_size=8,per_device_eval_batch_size=8, num_train_epochs=2):
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
        dataset = load_dataset(dataset_name)['train'].train_test_split(train_size=800, test_size=200)
    else:
        dataset = load_dataset(dataset_name)

    dataset = dataset.map(tokenize_dataset, batched=True)

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    class My_Callback_Transformers(TrainerCallback):
        def on_epoch_begin(self, args, state, control, **kwargs):
            self.start_time = time.time()
            self.forward_pass_time = 0

            self.batch_start_time = 0
            self.batch_end_time = 0
            self.batch_time = 0
            self.batches = 0

        def on_step_begin(self, args, state, control, **kwargs):
            self.batch_start_time = time.time()

        def on_step_end(self, args, state, control, **kwargs):
            self.batch_end_time = time.time()
            self.batch_time = self.batch_end_time - self.batch_start_time
            self.forward_pass_time += self.batch_time
            self.batches += 1

        def on_epoch_end(self, args, state, control, **kwargs):
            end_time = time.time()
            epoch_time = end_time - self.start_time
            fwd_pass_percent = round((self.forward_pass_time/epoch_time)*100,2)
            wgt_pass_time = epoch_time - self.forward_pass_time
            wgt_and_bkp_percent = round((wgt_pass_time/epoch_time)*100,2)
            print(f"\n  - Total:\t {round(epoch_time,10)}s")
            print(f"  - Forward:\t {round(self.forward_pass_time,10)}s ({fwd_pass_percent}%)")
            print(f"  - Wgt&BkP:\t {round(wgt_pass_time,10)}s ({wgt_and_bkp_percent}%)")
            print("Batches", self.batches)

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
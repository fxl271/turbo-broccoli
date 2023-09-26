from datasets import load_dataset, load_metric
from transformers import AutoTokenizer
from transformers import TrainingArguments, Trainer, TrainerCallback
from transformers import AutoModelForSequenceClassification
from transformers import pipeline
import types
import numpy as np

# code from https://medium.com/nlplanet/bert-finetuning-with-hugging-face-and-training-visualizations-with-tensorboard-46368a57fc97
# trains BERT model for sentiment classification of IMDb reviews
dataset_size = 80
split = 0.2


def preprocess_function_batch(examples):
    return tokenizer(examples["text"], truncation=True)


model_checkpoint = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(
    model_checkpoint, use_fast=True, model_max_length=512
)

dataset = load_dataset("imdb")
dataset = dataset["train"].select(np.random.permutation(25000)[:dataset_size])
splitted_datasets = dataset.train_test_split(test_size=split)
model_output_dir = "model_output"

metric = load_metric("accuracy")

splitted_datasets_encoded = splitted_datasets.map(
    preprocess_function_batch, batched=True
)

model = AutoModelForSequenceClassification.from_pretrained(
    model_checkpoint, num_labels=2
)

args = TrainingArguments(
    output_dir=model_output_dir,
    evaluation_strategy="steps",
    eval_steps=50,
    logging_strategy="steps",
    logging_steps=50,
    save_strategy="steps",
    save_steps=200,
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=1,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
)


def model_init():
    return AutoModelForSequenceClassification.from_pretrained(
        model_checkpoint, num_labels=2
    )


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    results = metric.compute(predictions=predictions, references=labels)
    return results


class MyCallback(TrainerCallback):
    "A callback that prints a message at every epoch"

    def on_epoch_begin(self, args, state, control, **kwargs):
        print("next epoch")


trainer = Trainer(
    model_init=model_init,
    args=args,
    train_dataset=splitted_datasets_encoded["train"],
    eval_dataset=splitted_datasets_encoded["test"]
    .shuffle(42)
    .select(range(int(split * dataset_size))),
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
    callbacks=[MyCallback()],
)

# trainer.train()
# trainer.save_model("/home/samuel/Documents/classes/csds_395/model_trained")
model = AutoModelForSequenceClassification.from_pretrained(
    "/home/samuel/Documents/classes/csds_395/model_trained"
)

MAX_LENGTH = model.config.max_position_embeddings


# need this in order truncate the inputs
def _my_preprocess(self, inputs, return_tensors=None, **preprocess_parameters):
    if return_tensors is None:
        return_tensors = self.framework
    model_inputs = self.tokenizer(
        inputs, truncation=True, max_length=MAX_LENGTH, return_tensors=return_tensors
    )
    return model_inputs


pipe = pipeline("text-classification", model=model, tokenizer=tokenizer)
pipe.preprocess = types.MethodType(_my_preprocess, pipe)

inputs = list(map(lambda x: x["text"], splitted_datasets["test"]))
# print(inputs)
results = pipe(inputs)
print(results)

test_references = np.array(splitted_datasets["test"]["label"])
metrics = metric.compute(
    predictions=list(map(lambda x: int(x["label"][6]), results)),
    references=test_references,
)
print(metrics)

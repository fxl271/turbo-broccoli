import datasets
import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from transformers import AutoModelForCausalLM
from transformers import AutoModelForMaskedLM
from transformers import AutoModelForTokenClassification
from transformers import DataCollatorForTokenClassification
from transformers import AutoModelForQuestionAnswering
from transformers import default_data_collator
from transformers import TrainingArguments
from transformers import TrainerCallback
from transformers import Trainer
from transformers import AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer
import time

#Xochi's callback function
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


#TEXT
#text classification
####################################################################################################
def train_text_classification(model_name, #MANDATORY, MUST BE STRING
                              dataset_name, #MANDATORY, MUST BE STRING
                              subset, #OPTIONAL, STRING OR NONE
                              label_names, #OPTIONAL, LIST OF STRINGS OR NONE*
                              # If None is provided, assumes all features of value ClassLabel are 
                              # labels
                              # If there are no features of value ClassLabel, defaults to "label" 
                              # feature
                              keys, #OPTIONAL, STRING (for one key) OR TUPLE (for two keys) OR NONE*
                              # If None is provided, assumes the first (or first two) feature(s) 
                              # with value type 'string' are keys
                              train_size, #MANDATORY, MUST BE INT
                              test_size, #MANDATORY, MUST BE INT
                              ):
    if subset == None:
        dataset = load_dataset(dataset_name)
    else:
        dataset = load_dataset(dataset_name, subset)

    dataset = dataset['train'].train_test_split(train_size=train_size, test_size=test_size)

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    if label_names == None:
        label_names = []
        for feature in dataset['train'].features:
            if type(dataset['train'].features[feature]) == datasets.features.features.ClassLabel:
                label_names.append(feature)
        if len(label_names) == 0:
            label_names.append('label')

    if keys == None:
        key1 = None
        key2 = None
        for feature in dataset['train'].features:
            if (key1 == None or key2 == None) and dataset['train'].features[feature].dtype == 'string':
                if key1 == None:
                    key1 = feature
                else:
                    key2 = feature
        if key2 == None:
            keys = key1
        else:
            keys = (key1, key2)

    # If there's more than 1 label name
    if len(label_names) != 1:
        # num_labels = the number of label names
        num_labels = len(label_names) 
    else:
        if type(dataset['train'].features[label_names[0]]) == datasets.features.features.ClassLabel: 
            # If the one label is a ClassLabel, num_labels = how many values it has
            num_labels = len(dataset['train'].features[label_names[0]].names)
        else:
            # If the one label is not a ClassLabel, num_labels = 1
            num_labels = 1
        
    print("Keys:", keys)
    print("Labels:", label_names)
    print("Number of Labels:", num_labels)

    for feature in label_names:
        # If a label is boolean, convert it to int
        if type(dataset['train'].features[feature] == datasets.features.features.Value) and (dataset['train'].features[feature].dtype == 'bool'):
            new_column = [int(x == True) for x in dataset['train'][feature]]
            dataset['train'] = dataset['train'].add_column("temp", new_column)
            dataset['train'] = dataset['train'].remove_columns(feature)
            dataset['train'] = dataset['train'].rename_column("temp", feature)
            new_column = [int(x == True) for x in dataset['test'][feature]]
            dataset['test'] = dataset['test'].add_column("temp", new_column)
            dataset['test'] = dataset['test'].remove_columns(feature)
            dataset['test'] = dataset['test'].rename_column("temp", feature)
        # If a label is long, convert it to float
        if type(dataset['train'].features[feature] == datasets.features.features.Value) and (dataset['train'].features[feature].dtype == 'int32'):
            new_column = [float(x) for x in dataset['train'][feature]]
            dataset['train'] = dataset['train'].add_column("temp", new_column)
            dataset['train'] = dataset['train'].remove_columns(feature)
            dataset['train'] = dataset['train'].rename_column("temp", feature)
            new_column = [float(x == True) for x in dataset['test'][feature]]
            dataset['test'] = dataset['test'].add_column("temp", new_column)
            dataset['test'] = dataset['test'].remove_columns(feature)
            dataset['test'] = dataset['test'].rename_column("temp", feature)
    
    # If the one label is not 'label'
    if num_labels == 1 and label_names[0] != 'label':
        if 'label' in dataset['train'].features:
            # Remove 'label' if it exists
            dataset['train'] = dataset['train'].remove_columns('label')
            dataset['test'] = dataset['test'].remove_columns('label')
        # Rename the one label to 'label'
        dataset['test'] = dataset['test'].rename_column(label_names[0], 'label')
        dataset['train'] = dataset['train'].rename_column(label_names[0], 'label')
        label_names[0] = 'label'

    def preprocess(examples):
        if type(keys) == tuple:
            tokens = tokenizer(examples[keys[0]], examples[keys[1]], truncation=True)
            num_tokens = len(examples[keys[0]])
        else:
            tokens = tokenizer(examples[keys], truncation=True)
            num_tokens = len(examples[keys])
        
        # If there's more than one label, make a labels matrix
        if (num_labels > 1):
            labels_batch = {k: examples[k] for k in examples.keys() if k in label_names}
            labels_matrix = np.zeros((num_tokens, num_labels))
            for idx, label in enumerate(label_names):
                labels_matrix[:, idx] = labels_batch[label]
            tokens["labels"] = labels_matrix.tolist()
        return tokens
    
    encoded_dataset = dataset.map(preprocess, batched=True)

    if len(label_names) == 1:
        model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
    else:
        model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels, problem_type="multi_label_classification")

    args = TrainingArguments("test-trainer",
                                    evaluation_strategy = "epoch",
                                    learning_rate=2e-5,
                                    per_device_train_batch_size=16,
                                    per_device_eval_batch_size=16,
                                    num_train_epochs=3,
                                    weight_decay=0.01)

    trainer = Trainer(
        model,
        args,
        train_dataset=encoded_dataset["train"],
        eval_dataset=encoded_dataset["test"],
        tokenizer=tokenizer,
        callbacks=[My_Callback_Transformers]
    )
    
    trainer.train()

#casual language modeling
####################################################################################################
def train_language_model_casual(model_name, #MANDATORY, MUST BE STRING
                              dataset_name, #MANDATORY, MUST BE STRING
                              block_size, #MANDATORY, MUST BE INT
                              subset, #OPTIONAL, STRING OR NONE
                              key, #OPTIONAL, STRING OR NONE*
                              # If None is provided, assumes the first feature with value type
                              # 'string' is the key
                              train_size, #MANDATORY, MUST BE INT
                              test_size, #MANDATORY, MUST BE INT
                              ):
    if subset == None:
        dataset = load_dataset(dataset_name)
    else:
        dataset = load_dataset(dataset_name, subset)

    dataset = dataset['train'].train_test_split(train_size=train_size, test_size=test_size)
    

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    if key == None:
        for feature in dataset['train'].features:
            if dataset['train'].features[feature].dtype == 'string':
                key = feature
                break

    print("Key:", key)

    def preprocess(examples): 
        tokens = tokenizer(examples[key])
        return tokens
    
    encoded_dataset = dataset.map(preprocess, batched=True, remove_columns=dataset['train'].column_names)

    def group_texts(examples):
        # Concatenate all texts.
        concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
            # customize this part to your needs.
        total_length = (total_length // block_size) * block_size
        # Split by chunks of max_len.
        result = {
            k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result
    
    encoded_dataset = encoded_dataset.map(group_texts, batched=True)

    model = AutoModelForCausalLM.from_pretrained(model_name)

    args = TrainingArguments("test-trainer",
                                    evaluation_strategy = "epoch",
                                    learning_rate=2e-5,
                                    per_device_train_batch_size=16,
                                    per_device_eval_batch_size=16,
                                    num_train_epochs=3,
                                    weight_decay=0.01)

    trainer = Trainer(
        model,
        args,
        train_dataset=encoded_dataset["train"],
        eval_dataset=encoded_dataset["test"],
        callbacks=[My_Callback_Transformers]
    )
    
    trainer.train()

# masked language modeling
####################################################################################################
def train_language_model_masked(model_name, #MANDATORY, MUST BE STRING
                              dataset_name, #MANDATORY, MUST BE STRING
                              block_size, #MANDATORY, MUST BE INT
                              subset, #OPTIONAL, STRING OR NONE
                              key, #OPTIONAL, STRING OR NONE*
                              # If None is provided, assumes the first feature with value type
                              # 'string' is the key
                              train_size, #MANDATORY, MUST BE INT
                              test_size, #MANDATORY, MUST BE INT
                              ):
    if subset == None:
        dataset = load_dataset(dataset_name)
    else:
        dataset = load_dataset(dataset_name, subset)

    
    dataset = dataset['train'].train_test_split(train_size=train_size, test_size=test_size)
    

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    if key == None:
        for feature in dataset['train'].features:
            if dataset['train'].features[feature].dtype == 'string':
                key = feature
                break

    print("Key:", key)

    def preprocess(examples): 
        tokens = tokenizer(examples[key])
        return tokens
    
    encoded_dataset = dataset.map(preprocess, batched=True, remove_columns=dataset['train'].column_names)

    def group_texts(examples):
        # Concatenate all texts.
        concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
            # customize this part to your needs.
        total_length = (total_length // block_size) * block_size
        # Split by chunks of max_len.
        result = {
            k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result

    encoded_dataset = encoded_dataset.map(group_texts, batched=True)

    model = AutoModelForMaskedLM.from_pretrained(model_name)
    
    args = TrainingArguments("test-trainer",
                                    evaluation_strategy = "epoch",
                                    learning_rate=2e-5,
                                    per_device_train_batch_size=16,
                                    per_device_eval_batch_size=16,
                                    num_train_epochs=3,
                                    weight_decay=0.01)

    trainer = Trainer(
        model,
        args,
        train_dataset=encoded_dataset["train"],
        eval_dataset=encoded_dataset["test"],
        callbacks=[My_Callback_Transformers]
    )
    
    trainer.train()

#token classification
####################################################################################################
def train_token_classification(model_name, #MANDATORY, MUST BE STRING
                              dataset_name, #MANDATORY, MUST BE STRING
                              label_all_tokens, #MANDATORY, MUST BE BOOLEAN
                              subset, #OPTIONAL, STRING OR NONE
                              key, #OPTIONAL, STRING OR NONE*
                              # If None is provided, assumes the first feature with value type
                              # 'string', or the first feature of value Sequence containing features
                              # of value type 'string' is the key
                              tags, #OPTIONAL, STRING OR NONE*
                              # If None is provided, assumes the first feature of value Sequence
                              # continue features of value ClassLabel is the tags feature
                              train_size, #MANDATORY, MUST BE INT
                              test_size, #MANDATORY, MUST BE INT
                              ):
    if subset == None:
        dataset = load_dataset(dataset_name)
    else:
        dataset = load_dataset(dataset_name, subset)

    dataset = dataset['train'].train_test_split(train_size=train_size, test_size=test_size)

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    if key == None:
        for feature in dataset['train'].features:
            if dataset['train'].features[feature].dtype == 'string' or type(dataset['train'].features[feature]) == datasets.features.features.Sequence and dataset['train'].features[feature].feature.dtype == 'string':
                key = feature
                break
    
    if tags == None:
        for feature in dataset['train'].features:
            if type(dataset['train'].features[feature]) == datasets.features.features.Sequence and type(dataset['train'].features[feature].feature) == datasets.features.features.ClassLabel:
                tags = feature
                break

    label_list = dataset["train"].features[tags].feature.names

    
    print("Key:", key)
    print("Tags:", tags)
    print("Label List:", label_list)
    

    def preprocess(examples):
        tokens = tokenizer(examples[key], truncation=True, is_split_into_words=(type(dataset['train'].features[tags]) == datasets.features.features.Sequence))

        labels = []
        for i, label in enumerate(examples[tags]):
            word_ids = tokens.word_ids(batch_index=i)
            previous_word_idx = None
            label_ids = []
            for word_idx in word_ids:
                # Special tokens have a word id that is None. We set the label to -100 so they are automatically
                # ignored in the loss function.
                if word_idx is None:
                    label_ids.append(-100)
                # We set the label for the first token of each word.
                elif word_idx != previous_word_idx:
                    label_ids.append(label[word_idx])
                # For the other tokens in a word, we set the label to either the current label or -100, depending on
                # the label_all_tokens flag.
                else:
                    label_ids.append(label[word_idx] if label_all_tokens else -100)
                previous_word_idx = word_idx

            labels.append(label_ids)

        tokens["labels"] = labels
        return tokens
    
    encoded_dataset = dataset.map(preprocess, batched=True)

    model = AutoModelForTokenClassification.from_pretrained(model_name, num_labels=len(label_list))

    args = TrainingArguments("test-trainer",
                                    evaluation_strategy = "epoch",
                                    learning_rate=2e-5,
                                    per_device_train_batch_size=16,
                                    per_device_eval_batch_size=16,
                                    num_train_epochs=3,
                                    weight_decay=0.01)
    
    data_collator = DataCollatorForTokenClassification(tokenizer)

    trainer = Trainer(
        model,
        args,
        train_dataset=encoded_dataset["train"],
        eval_dataset=encoded_dataset["test"],
        data_collator=data_collator,
        callbacks=[My_Callback_Transformers]
    )
    
    trainer.train()

#extractive question answering
####################################################################################################
def train_extractive_qa(model_name, #MANDATORY, MUST BE STRING
                        dataset_name, #MANDATORY, MUST BE STRING
                        max_length, #MANDATORY, MUST BE INT
                        doc_stride, #MANDATORY, MUST BE INT
                        subset, #OPTIONAL, STRING OR NONE
                        context, #OPTIONAL, STRING OR NONE*
                        # If None is provided, assumes the first feature with value type 'string' 
                        # is the context
                        question, #OPTIONAL, STRING OR NONE*
                        # If None is provided, assumes the first feature (or second feature, if the
                        # first feature is the context) with value type 'string' is the question
                        answer, #OPTIONAL, STRING OR NONE*
                        # If None is provided, assumes the first feature that aligns with the answer
                        # format of the SQUAD datasets (sequence with keys text & answer_start) is
                        # the answer
                        train_size, #MANDATORY, MUST BE INT
                        test_size, #MANDATORY, MUST BE INT
                        ):
    if subset == None:
        dataset = load_dataset(dataset_name)
    else:
        dataset = load_dataset(dataset_name, subset)
    
    
    dataset = dataset['train'].train_test_split(train_size=train_size, test_size=test_size)
    

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    pad_on_right = tokenizer.padding_side == "right"

    if context == None:
        for feature in dataset['train'].features:
            if dataset['train'].features[feature].dtype == 'string':
                context = feature
                break

    if question == None:
        for feature in dataset['train'].features:
            if dataset['train'].features[feature].dtype == 'string' and feature != context:
                question = feature
                break

    if answer == None:
        for feature in dataset['train'].features:
            if (type(dataset['train'].features[feature]) == datasets.features.features.Sequence and type(dataset['train'].features[feature].feature) is dict and 'text' in dataset['train'].features[feature].feature.keys() and 'answer_start' in dataset['train'].features[feature].feature.keys()) or (type(dataset['train'].features[feature]) is dict and 'text' in dataset['train'].features[feature].keys() and 'answer_start' in dataset['train'].features[feature].keys()):
                answer = feature
                break
    
    
    print("Context:", context)
    print("Question:", question)
    print("Answer:", answer)
    


    def prepare_train_features(examples):
        # Some of the questions have lots of whitespace on the left, which is not useful and will make the
        # truncation of the context fail (the tokenized question will take a lots of space). So we remove that
        # left whitespace
        examples[question] = [q.lstrip() for q in examples[question]]

        # Tokenize our examples with truncation and padding, but keep the overflows using a stride. This results
        # in one example possible giving several features when a context is long, each of those features having a
        # context that overlaps a bit the context of the previous feature.
        tokenized_examples = tokenizer(
            examples[question if pad_on_right else context],
            examples[context if pad_on_right else question],
            truncation="only_second" if pad_on_right else "only_first",
            max_length=max_length,
            stride=doc_stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length",
        )

        # Since one example might give us several features if it has a long context, we need a map from a feature to
        # its corresponding example. This key gives us just that.
        sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")
        # The offset mappings will give us a map from token to character position in the original context. This will
        # help us compute the start_positions and end_positions.
        offset_mapping = tokenized_examples.pop("offset_mapping")

        # Let's label those examples!
        tokenized_examples["start_positions"] = []
        tokenized_examples["end_positions"] = []

        for i, offsets in enumerate(offset_mapping):
            # We will label impossible answers with the index of the CLS token.
            input_ids = tokenized_examples["input_ids"][i]
            cls_index = input_ids.index(tokenizer.cls_token_id)

            # Grab the sequence corresponding to that example (to know what is the context and what is the question).
            sequence_ids = tokenized_examples.sequence_ids(i)

            # One example can give several spans, this is the index of the example containing this span of text.
            sample_index = sample_mapping[i]
            answers = examples[answer][sample_index]
            # If no answers are given, set the cls_index as answer.
            if len(answers["answer_start"]) == 0:
                tokenized_examples["start_positions"].append(cls_index)
                tokenized_examples["end_positions"].append(cls_index)
            else:
                # Start/end character index of the answer in the text.
                start_char = answers["answer_start"][0]
                end_char = start_char + len(answers["text"][0])

                # Start token index of the current span in the text.
                token_start_index = 0
                while sequence_ids[token_start_index] != (1 if pad_on_right else 0):
                    token_start_index += 1

                # End token index of the current span in the text.
                token_end_index = len(input_ids) - 1
                while sequence_ids[token_end_index] != (1 if pad_on_right else 0):
                    token_end_index -= 1

                # Detect if the answer is out of the span (in which case this feature is labeled with the CLS index).
                if not (offsets[token_start_index][0] <= start_char and offsets[token_end_index][1] >= end_char):
                    tokenized_examples["start_positions"].append(cls_index)
                    tokenized_examples["end_positions"].append(cls_index)
                else:
                    # Otherwise move the token_start_index and token_end_index to the two ends of the answer.
                    # Note: we could go after the last offset if the answer is the last word (edge case).
                    while token_start_index < len(offsets) and offsets[token_start_index][0] <= start_char:
                        token_start_index += 1
                    tokenized_examples["start_positions"].append(token_start_index - 1)
                    while offsets[token_end_index][1] >= end_char:
                        token_end_index -= 1
                    tokenized_examples["end_positions"].append(token_end_index + 1)

        return tokenized_examples
    
    tokenized_datasets = dataset.map(prepare_train_features, batched=True, remove_columns=dataset["train"].column_names)
    model = AutoModelForQuestionAnswering.from_pretrained(model_name)
    args = TrainingArguments("test-trainer",
                                    evaluation_strategy = "epoch",
                                    learning_rate=2e-5,
                                    per_device_train_batch_size=16,
                                    per_device_eval_batch_size=16,
                                    num_train_epochs=3,
                                    weight_decay=0.01)
    data_collator = default_data_collator
    trainer = Trainer(
        model,
        args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["test"],
        data_collator=data_collator,
        callbacks=[My_Callback_Transformers]
    )
    
    trainer.train()

#translation
####################################################################################################
def train_translation(model_name, #MANDATORY, MUST BE STRING
                        dataset_name, #MANDATORY, MUST BE STRING
                        max_input_length, #MANDATORY, MUST BE INT
                        max_target_length, #MANDATORY, MUST BE INT
                        src_lang, #MANDATORY, MUST BE STRING
                        target_lang, #MANDATORY, MUST BE 
                        tokenizer_src, #OPTIONAL, STRING OR NONE
                        tokenizer_tgt, #OPTIONAL, STRING OR NONE
                        subset, #OPTIONAL, STRING OR NONE
                        prefix, #OPTIONAL, STRING OR NONE
                        trans_pair, #OPTIONAL, STRING OR NONE*
                        # If None is provided, assumes the first feature with Translation or dict
                        # with relevant features is the translation pair. If none are found, it
                        # attempts to run with src_lang and target_lang as features instead of keys
                        train_size, #MANDATORY, MUST BE INT
                        test_size, #MANDATORY, MUST BE INT
                        ):
    if prefix == None:
        prefix = ""

    if subset == None:
        dataset = load_dataset(dataset_name)
    else:
        dataset = load_dataset(dataset_name, subset)
    
    
    dataset = dataset['train'].train_test_split(train_size=train_size, test_size=test_size)
    

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    if tokenizer_src:
        tokenizer.src_lang = tokenizer_src
    
    if tokenizer_tgt:
        tokenizer.tgt_lang = tokenizer_tgt

    if trans_pair == None:
        for feature in dataset['train'].features:
            if (type(dataset['train'].features[feature]) == datasets.features.features.Translation) or (type(dataset['train'].features[feature]) == datasets.features.features.Sequence and type(dataset['train'].features[feature].feature) is dict and src_lang in dataset['train'].features[feature].feature.keys() and target_lang in dataset['train'].features[feature].feature.keys()) or (type(dataset['train'].features[feature]) is dict and src_lang in dataset['train'].features[feature].keys() and target_lang in dataset['train'].features[feature].keys()):
                trans_pair = feature
                break

    
    print("Translation Pair:", trans_pair)
    

    def preprocess(examples):
        if trans_pair:
            inputs = [prefix + ex[src_lang] for ex in examples[trans_pair]]
            targets = [ex[target_lang] for ex in examples[trans_pair]]
        else:
            inputs = [prefix + in_src for in_src in examples[src_lang]]
            targets = examples[target_lang]
        tokens = tokenizer(inputs, max_length=max_input_length, truncation=True)
        # Setup the tokenizer for targets
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(targets, max_length=max_target_length, truncation=True)

        tokens["labels"] = labels["input_ids"]
        return tokens
    
    encoded_dataset = dataset.map(preprocess, batched=True)

    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    
    args = Seq2SeqTrainingArguments("test-trainer",
                                    evaluation_strategy = "epoch",
                                    learning_rate=2e-5,
                                    per_device_train_batch_size=16,
                                    per_device_eval_batch_size=16,
                                    num_train_epochs=3,
                                    weight_decay=0.01)

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    trainer = Seq2SeqTrainer(
        model,
        args,
        train_dataset=encoded_dataset["train"],
        eval_dataset=encoded_dataset["test"],
        data_collator=data_collator,
        tokenizer=tokenizer,
        callbacks=[My_Callback_Transformers]
    )
    
    trainer.train()

#summarization
####################################################################################################
def train_summarization(model_name, #MANDATORY, MUST BE STRING
                        dataset_name, #MANDATORY, MUST BE STRING
                        add_prefix, #MANDATORY, MUST BE BOOLEAN
                        max_input_length, #MANDATORY, MUST BE INT
                        max_target_length, #MANDATORY, MUST BE INT
                        subset, #OPTIONAL, STRING OR NONE
                        document, #OPTIONAL, STRING OR NONE*
                        # If None is provided, assumes the first feature with value type 'string' 
                        # is the document
                        summary, #OPTIONAL, STRING OR NONE*
                        # If None is provided, assumes the first feature (or second feature, if the
                        # first feature is the summary) with value type 'string' is the question
                        train_size, #MANDATORY, MUST BE INT
                        test_size, #MANDATORY, MUST BE INT
                        ):
    if subset == None:
        dataset = load_dataset(dataset_name)
    else:
        dataset = load_dataset(dataset_name, subset)

    
    dataset = dataset['train'].train_test_split(train_size=train_size, test_size=test_size)
    

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    if document == None:
        for feature in dataset['train'].features:
            if dataset['train'].features[feature].dtype == 'string':
                document = feature
                break

    if summary == None:
        for feature in dataset['train'].features:
            if dataset['train'].features[feature].dtype == 'string' and feature != document:
                summary = feature
                break

    
    print("Document:", document)
    print("Summary:", summary)
    

    def preprocess(examples):
        inputs = examples[document]
        if add_prefix:
            inputs = ["summarize: " + doc for doc in inputs]
        tokens = tokenizer(inputs, max_length=max_input_length, truncation=True)

        # Setup the tokenizer for targets
        labels = tokenizer(text_target=examples[summary], max_length=max_target_length, truncation=True)

        tokens["labels"] = labels["input_ids"]
        return tokens
    
    encoded_dataset = dataset.map(preprocess, batched=True)

    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    
    args = Seq2SeqTrainingArguments("test-trainer",
                                    evaluation_strategy = "epoch",
                                    learning_rate=2e-5,
                                    per_device_train_batch_size=16,
                                    per_device_eval_batch_size=16,
                                    num_train_epochs=3,
                                    weight_decay=0.01)

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    trainer = Seq2SeqTrainer(
        model,
        args,
        train_dataset=encoded_dataset["train"],
        eval_dataset=encoded_dataset["test"],
        data_collator=data_collator,
        tokenizer=tokenizer,
        callbacks=[My_Callback_Transformers]
    )
    
    trainer.train()

#######DEBUGGING LINES, REMOVE AFTER TESTING#######
#train_summarization("t5-small", "samsum", add_prefix = True, max_input_length = 1024, max_target_length = 128, subset = None, document = "dialogue", summary = "summary", train_size=800, test_size=200)
#train_summarization("t5-small", "EdinburghNLP/xsum", add_prefix = True, max_input_length = 1024, max_target_length = 128, subset = None, document = None, summary = None, train_size=800, test_size=200)
#train_text_classification("distilbert-base-uncased", "glue", subset = "qqp", label_names = None, keys = ("question2", "question1"), train_size=800, test_size=200)  #2 keys, 1 label (2 int values, manually set)
#train_text_classification("distilbert-base-uncased", "fhamborg/news_sentiment_newsmtsc", subset = "mt", keys = None, label_names = ["polarity"], train_size=800, test_size=200) #1 key, 1 label (1 long value)
#train_text_classification("distilbert-base-uncased", "allocine", subset = None, keys = None, label_names = None, train_size=800, test_size=200)  #1 key, 1 label (2 int values)
#train_text_classification("distilbert-base-uncased", "glue", subset = "stsb", label_names = None, keys = None, train_size=800, test_size=200) #1 key, 1 label (float)
#train_text_classification("distilbert-base-uncased", "financial_phrasebank", subset = "sentences_50agree", label_names = None, keys = None, train_size=800, test_size=200) #1 key, 1 label (3 int values)
#train_text_classification("distilbert-base-uncased", "glue", subset = "cola", label_names = None, keys = None, train_size=800, test_size=200) #1 key, 1 label (2 int values)
#train_text_classification("distilbert-base-uncased", "rotten_tomatoes", subset = None, label_names = None, keys = None, train_size=800, test_size=200) #1 key, 1 label (2 int values)
#train_text_classification("distilbert-base-uncased", "glue", subset = "qqp", label_names = None, keys = None, train_size=800, test_size=200)  #2 keys, 1 label (2 int values)
#train_text_classification("bert-base-uncased", "sem_eval_2018_task_1", subset = "subtask5.english", keys = "Tweet", label_names=['anger', 'anticipation', 'disgust', 'fear', 'joy', 'love', 'optimism', 'pessimism', 'sadness', 'surprise', 'trust'], train_size=800, test_size=200)  #1 key, 11 labels (boolean)
#train_translation("Helsinki-NLP/opus-mt-en-ro", "phongmt184172/mtet", max_input_length = 128, max_target_length = 128, src_lang = "source", target_lang = "target", tokenizer_src=None, tokenizer_tgt=None, subset=None, prefix=None, trans_pair=None)    
#train_translation("Helsinki-NLP/opus-mt-en-ro", "chr_en", max_input_length = 128, max_target_length = 128, src_lang = "chr", target_lang = "en", tokenizer_src=None, tokenizer_tgt=None, subset="parallel", prefix=None, trans_pair=None, train_size=800, test_size=200)
#train_translation("Helsinki-NLP/opus-mt-en-ro", "aslg_pc12", max_input_length = 128, max_target_length = 128, src_lang = "gloss", target_lang = "text", tokenizer_src=None, tokenizer_tgt=None, subset=None, prefix=None, trans_pair=None, train_size=800, test_size=200)
#train_extractive_qa("distilbert-base-uncased", "covid_qa_deepset", max_length = 384, doc_stride = 128, subset = None, context = None, question = None, answer = None, train_size=800, test_size=200)
#train_extractive_qa("distilbert-base-uncased", "adversarial_qa", max_length = 384, doc_stride = 128, subset = "adversarialQA", context = "context", question = "question", answer = None, train_size=800, test_size=200)
#train_extractive_qa("distilbert-base-uncased", "squad_v2", max_length = 384, doc_stride = 128, subset = None, context = "context", question = "question", answer = None, train_size=800, test_size=200)
#train_token_classification("distilbert-base-uncased", "conll2003", label_all_tokens = True, subset = None, key = "tokens", tags = "ner_tags", train_size=800, test_size=200)
#train_language_model_casual("distilgpt2", "wikitext", block_size = 128, subset = "wikitext-2-raw-v1", key = "text", train_size=800, test_size=200)
#train_language_model_masked("distilroberta-base", "wikitext", block_size = 128, subset = "wikitext-2-raw-v1", key = "text", train_size=800, test_size=200)
###################################################
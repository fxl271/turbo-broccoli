from TrainApp import train_model, train_model_peft

# train_model("distilbert-base-uncased", "rotten_tomatoes")


class MLThing:
    def __init__(self):
        pass

    def setVars(self, mlModel, dataset, params, subset=None, peftType=None):
        self.mlModel = mlModel
        self.dataset = dataset
        self.params = params
        self.subset = subset
        self.peft_type = peftType

    def setLbArray(self, lbArray):
        self.lbArray = lbArray
        print(self.lbArray)

    def setPArray(self, pArray):
        self.pArray = pArray
        print(self.pArray)

    def run(self):
        if self.peftType is not None:
            if self.subset is not None:
                train_model_peft(
                    model_name=self.mlModel,
                    dataset_name=self.dataset,
                    limit_size=eval(self.params[2]),
                    output_dir="path/to/save/folder/",
                    learning_rate=self.params[1],
                    per_device_train_batch_size=self.params[3],
                    per_device_eval_batch_size=self.params[4],
                    num_train_epochs=self.params[0],
                    subset_name=self.subset,
                    peftType=self.peft_type,
                )
            else:
                train_model_peft(
                    model_name=self.mlModel,
                    dataset_name=self.dataset,
                    limit_size=eval(self.params[2]),
                    output_dir="path/to/save/folder/",
                    learning_rate=self.params[1],
                    per_device_train_batch_size=self.params[3],
                    per_device_eval_batch_size=self.params[4],
                    num_train_epochs=self.params[0],
                    peftType=self.peft_type,
                )
        else:
            if self.subset is not None:
                train_model(
                    model_name=self.mlModel,
                    dataset_name=self.dataset,
                    limit_size=eval(self.params[2]),
                    output_dir="path/to/save/folder/",
                    learning_rate=self.params[1],
                    per_device_train_batch_size=self.params[3],
                    per_device_eval_batch_size=self.params[4],
                    num_train_epochs=self.params[0],
                    subset_name=self.subset,
                )
            else:
                train_model(
                    model_name=self.mlModel,
                    dataset_name=self.dataset,
                    limit_size=eval(self.params[2]),
                    output_dir="path/to/save/folder/",
                    learning_rate=self.params[1],
                    per_device_train_batch_size=self.params[3],
                    per_device_eval_batch_size=self.params[4],
                    num_train_epochs=self.params[0],
                )
        # train_model("bigscience/bloomz-560m", "bigscience/xP3")


# https://huggingface.co/distilbert-base-uncased
# https://huggingface.co/datasets/rotten_tomatoes

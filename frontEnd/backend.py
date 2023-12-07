from TrainApp import train_model, train_model_peft
from newTrainApp import *
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

    def setLbArray(self, lbArray, trainingType):
        self.lbArray = lbArray
        self.trainingType = trainingType
        print(self.lbArray) #Format [[labelname0, labeldatatype0], [labelname1, labeldatatype1]]
        print(self.trainingType) #Format [index in list, "name of trainingtype"]
        
    def setPArray(self, pArray):
        self.pArray = pArray
        print(self.pArray) #Format [val0, val1]

    def run2(self):
        print("Run with this shit") 
        print(self.mlModel)
        print(self.dataset)
        print(self.subset)
        print(self.pArray[0])        
        print(self.pArray[1])
        print(self.pArray[2])
        print(self.pArray[3])   
        
        for i in range(0, len(self.pArray)):
            if (self.lbArray[i][1] == "int"):
                self.pArray[i] = int(self.pArray[i])
            if (self.lbArray[i][1] == "bool"):
                self.pArray[i] = bool(self.pArray[i])
            
                
        print(type(self.pArray[0]))        
        print(type(self.pArray[1]))
        print(type(self.pArray[2]))
        print(type(self.pArray[3]))   
        
        
        #train_text_classification("distilbert-base-uncased", "glue", subset = "qqp", label_names = None, keys = ("question2", "question1"), train_size=800, test_size=200)  #2 keys, 1 label (2 int values, manually set)

        match (self.trainingType[0]):
            case 0:
                #Train_Text_Classification
                
                #Deal with tuple here
                keyValue = None
                if (type(self.pArray[1]) is list):
                    if (len(self.pArray[1]) == 1):
                        keyValue = self.pArray[1][0]
                    else:
                        if (len(self.pArray[1]) >= 2):
                            keyValue = (self.pArray[1][0], self.pArray[1][1])
                    
                train_text_classification(
                    model_name=self.mlModel,
                    dataset_name=self.dataset,
                    subset=self.subset,
                    label_names=self.pArray[0],
                    keys=keyValue,
                    train_size=self.pArray[2],
                    test_size=self.pArray[3]
                )
            case 1:
                #train_language_model_casual
                
                train_language_model_casual(
                    model_name=self.mlModel,
                    dataset_name=self.dataset,
                    subset=self.subset,
                    block_size=self.pArray[0],
                    key=self.pArray[1],
                    train_size=self.pArray[2],
                    test_size=self.pArray[3]
                )
                
            case 2:
                #train_language_model_masked
                train_language_model_masked(
                    model_name=self.mlModel,
                    dataset_name=self.dataset,
                    subset=self.subset,
                    block_size=self.pArray[0],
                    key=self.pArray[1],
                    train_size=self.pArray[2],
                    test_size=self.pArray[3]
                )
                
            case 3:
                #train_token_classification
                train_token_classification(
                    model_name=self.mlModel,
                    dataset_name=self.dataset,
                    subset=self.subset,
                    label_all_tokens=self.pArray[0],
                    key=self.pArray[1],
                    tags=self.pArray[2],
                    train_size=self.pArray[3],
                    test_size=self.pArray[4]
                )
                
            case 4:
                #train_extractive_qa
                train_extractive_qa(
                    model_name=self.mlModel,
                    dataset_name=self.dataset,
                    subset=self.subset,
                    max_length=self.pArray[0],
                    doc_stride=self.pArray[1],
                    context=self.pArray[2],
                    question=self.pArray[3],
                    answer=self.pArray[4],
                    train_size=self.pArray[5],
                    test_size=self.pArray[6]
                )

            case 5:
                #train_translation
                train_translation(
                    model_name=self.mlModel,
                    dataset_name=self.dataset,
                    subset=self.subset,
                    max_input_length=self.pArray[0],
                    max_target_length=self.pArray[1],
                    src_lang=self.pArray[2],
                    target_lang=self.pArray[3],
                    tokenizer_src=self.pArray[4],
                    tokenizer_tgt=self.pArray[5],
                    prefix=self.pArray[6],
                    trans_pair=self.pArray[7],
                    train_size=self.pArray[8],
                    test_size=self.pArray[9]
                )
                
            case 6:
                #train_summarization
                train_summarization(
                    model_name=self.mlModel,
                    dataset_name=self.dataset,
                    subset=self.subset,
                    add_prefix=self.pArray[0],
                    max_input_length=self.pArray[1],
                    max_target_length=self.pArray[2],
                    document=self.pArray[3],
                    summary=self.pArray[4],
                    train_size=self.pArray[5],
                    test_size=self.pArray[6]
                )

    def run(self):
        if self.peft_type is not None:
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

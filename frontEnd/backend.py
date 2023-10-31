from TrainApp import train_model

#train_model("distilbert-base-uncased", "rotton_tomatos")

class MLThing:
    def __init__(self, mlModel, dataset, params):
        self.mlModel = mlModel
        self.dataset = dataset
        self.params = params
        
        #train_model(self.mlModel, self.dataset)
        
        train_model(model_name=self.mlModel, 
                    dataset_name=self.dataset, 
                    limit_size=eval(params[2]),
                    output_dir="path/to/save/folder/", 
                    learning_rate=self.params[1], 
                    per_device_train_batch_size=self.params[3], 
                    per_device_eval_batch_size=self.params[4], 
                    num_train_epochs=self.params[0])
        #train_model("bigscience/bloomz-560m", "bigscience/xP3")
    
    
#https://huggingface.co/distilbert-base-uncased
#https://huggingface.co/datasets/rotten_tomatoes

    
    


from TrainApp import train_model, train_model_peft

#train_model("distilbert-base-uncased", "rotton_tomatos")

class MLThing:
    def __init__(self, mlModel, dataset, params):
        self.mlModel = mlModel
        self.dataset = dataset
        self.params = params
        
        train_model(self.mlModel, self.dataset, self.params)
        #train_model("bigscience/bloomz-560m", "bigscience/xP3")
    
    
#https://huggingface.co/distilbert-base-uncased
#https://huggingface.co/datasets/rotten_tomatoes

    
    


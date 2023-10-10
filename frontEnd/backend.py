from TrainApp import train_model



class MLThing:
    def __init__(self, mlModel, dataset, params):
        self.mlModel = mlModel
        self.dataset = dataset
        self.params = params
        
        train_model(self.mlModel, self.dataset, self.params)
        
    
    
#https://huggingface.co/distilbert-base-uncased
#https://huggingface.co/datasets/rotten_tomatoes

    
    


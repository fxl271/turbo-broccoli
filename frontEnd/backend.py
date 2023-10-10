from TrainApp import train_model



class MLThing:
    def __train__(self):
        train_model(self.mlModel, self.dataset, self.params)
        
    def __init__(self, mlModel, dataset, params):
        self.mlModel = mlModel
        self.dataset = dataset
        self.params = params
        
        __train__(self.mlModel, self.dataset, self.params)
        
    
    
#https://huggingface.co/distilbert-base-uncased
#https://huggingface.co/datasets/rotten_tomatoes

    
    


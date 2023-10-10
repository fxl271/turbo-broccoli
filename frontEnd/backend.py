from TrainApp import train_model



class MLThing:
    def __train__(self):
        train_model(self.mlModel, self.dataset, self.params)
        
    def __init__(self, mlModel, dataset, params):
        self.mlModel = mlModel
        self.dataset = dataset
        self.params = params
        

    
    
    

    
    


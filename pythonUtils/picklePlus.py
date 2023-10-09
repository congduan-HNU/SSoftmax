import pickle
import os

def pickleSave(obj, fileName:str):
    if not os.path.exists(os.path.dirname(fileName)):
        os.makedirs(os.path.dirname(fileName))
    with open(fileName,'wb+') as f:
        pickle.dump(obj, f)
        
def pickRead(fileName:str):
    with open(fileName, 'rb') as f:
        return pickle.load(f)
from hummingbird.ml import convert
import pickle
import torch


rud_model = pickle.load(open("D:/Chinmay/ML Pipeline/Trained model/mode_1_20201006-083222", "rb"))
rud_model.verbose = False
print(rud_model.n_estimators, rud_model.max_depth, rud_model.max_features)
print("Model loaded successfully:: Now converting to hummingbird model")
rud_model = convert(rud_model, 'pytorch')
print("Converted Sklearn model to : ", type(rud_model))
torch.save(rud_model.state_dict(), "hummingbird_models/rud")



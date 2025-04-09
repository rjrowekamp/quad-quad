# -*- coding: utf-8 -*-
"""
Created on Wed Feb 14 15:28:38 2018

@author: rrowekamp
"""
import sys
from numpy import zeros
from sklearn.model_selection import train_test_split
from keras.models import Model

from analysis_tools import make_best_model
from stim_tools import load_stim

# Create a model with the named layer as the output
def make_submodel(model,layer_name):
        
    for layer in model.layers:
        if layer.name == layer_name:
            sublayer = layer
            break
            
    return Model(model.layers[0].input,sublayer.output)

# Calculate the variance of the linear and quadratic (if any) components of each layer
def main(dataset,cell,model_type):

    lin_var = zeros((2,4))
    quad_var = zeros((2,4))

    stim = load_stim(dataset,cell)[1]

    for seed in range(4):

        stim_train = train_test_split(stim,test_size=0.25,random_state=seed)[0]

        model = make_best_model(dataset,cell,model_type,seed)
    
        submodel = make_submodel(model,'L1')
    
        lin_var[0,seed] = submodel.predict(stim_train).var()
        
        submodel = make_submodel(model,'L2')
    
        lin_var[1,seed] = submodel.predict(stim_train).var()
    
        if model_type == 'QQ':
                            
            submodel = make_submodel(model,'dense')
            
            quad_var[0,seed] = submodel.predict(stim_train).var()
                            
            submodel = make_submodel(model, 'dense_1')
            
            quad_var[1,seed] = submodel.predict(stim_train).var()
            
        elif model_type == 'QL':
            
            submodel = make_submodel(model,'dense')
            
            quad_var[0,seed] = submodel.predict(stim_train).var()
            
        elif model_type == 'LQ':
            
            submodel = make_submodel(model,'dense')
            
            quad_var[1,seed] = submodel.predict(stim_train).var()
        
    return lin_var,quad_var

if __name__ == '__main__':
    
    dataset = int(sys.argv[1])

    cell = int(sys.argv[2])

    model_type = sys.argv[3]
    
    lin_var,quad_var = main(dataset,cell,model_type)
    
    filename = f'../Results/dataset_{dataset}/cell_{cell}/model-{model_type}_variance.dat'
   
    with open(filename,'wb') as f:
        lin_var.tofile(f)
        quad_var.tofile(f)
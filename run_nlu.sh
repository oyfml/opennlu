#!/bin/bash

## Training part 
[ ! -d "data_nlu" ]  && mkdir data_nlu

cp data/bus_data_train.md data_nlu/.


echo language: "en" > config.yml
echo "" >> config.yml
echo pipeline: "pretrained_embeddings_convert">> config.yml

rasa train nlu --nlu data_nlu

rm -rf data_nlu

## Testing part 
[ ! -d "data_nlu" ]  && mkdir data_nlu

cp data/bus_data_test.md data_nlu/.

rasa test nlu --nlu data_nlu

rm -rf data_nlu

## Save the model 
mv models data/model/rasa/bus_nlu_convert
rm -rf results config.yml
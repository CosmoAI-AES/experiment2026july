
#!/bin/sh

log=datagen.log 
err=datagen.err 

dataset.csv: dataset.toml 
	time python -m CosmoSim.dataset $< $@ 
roulette.csv: dataset.csv
	time python -m CosmoSim --toml dataset.toml \
             --csvfile dataset.csv --outfile roulette.csv  \
             --directory images -vv  > $(log) 2> $(err)

training.csv: roulette.csv
	time python -m droulette.split problem.toml



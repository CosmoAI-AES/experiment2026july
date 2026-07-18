#!/bin/sh

all: experiment001/eval.log

log=datagen.log 
err=datagen.err 

P=sie

$P-dataset.csv: $P-dataset.toml 
	time python -m CosmoSim.dataset $< $@ 

$P-roulette.csv: $P-dataset.csv
	time python -m CosmoSim --toml $P-dataset.toml \
             --csvfile $< --outfile $@ \
             --directory images -vv  > $(log) 2> $(err)

$P-training.csv: $P-problem.toml $P-roulette.csv
	time python -m droulette.split $<


experiment%/eval.log: experiment% training.csv
	time python -m droulette.eval -o $D/eval.csv $< > $@ 2> $</eval.err
	echo Done $<

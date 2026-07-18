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

experiment001: experimentbatch.toml
	python -m droulette.batch $<

experiment%/modeltraining.log: experiment%/ml.toml $P-training.csv
	time python -m droulette.model --config $<  \
	     > $@ 2> experiment$*/modeltraining.err

experiment%/eval.log: experiment%  experiment%/modeltraining.log
	time python -m droulette.eval -o $</eval.csv $< > $@ 2> $</eval.err
	echo Done $<

clean:
	rm -f *.csv
	rm -rf images resimulation

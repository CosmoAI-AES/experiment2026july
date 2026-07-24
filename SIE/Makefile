all: experiment001/eval.log

log=datagen.log 
err=datagen.err 

P=sie

data: $P-training.csv $P-testing.toml $P-validation.csv

$P-dataset.csv: $P-dataset.toml 
	time python -m CosmoSim.dataset $< $@ 

$P-roulette.csv: $P-dataset.csv
	time python -m CosmoSim --toml $P-dataset.toml \
             --csvfile $< --outfile $@ \
             --directory images -vv  > $(log) 2> $(err)

$P-training.csv: $P-problem.toml $P-roulette.csv
	time python -m droulette.split $<

%-testing.csv: %-training.csv
%-validation.csv: %-training.csv 

experiment001: experimentbatch.toml
	python -m droulette.batch $<

experiment%/model.pth: experiment%/ml.toml $P-training.csv
	time python -m droulette.model --config $< 

experiment%/eval.log: experiment%  experiment%/model.pt
	time python -m droulette.eval -o $</eval.csv $< > $@ 2> $</eval.err
	echo Done $<

clean:
	rm -f Dataset.ipynb
	cd experiment001 && $(MAKE) $@
distclean: clean
	rm -rf images resimulation
	rm -f */model.pth

ipynb: Dataset.ipynb
	cd experiment001 && $(MAKE) $@
J=jupytext --execute -w --to ipynb

%.ipynb: %.md
	$J "$<"

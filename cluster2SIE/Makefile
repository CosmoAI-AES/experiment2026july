all: experiment002/model.pth

log=datagen.log 
err=datagen.err 

P=cluster2

$P-dataset.csv: $P-dataset.toml 
	time python -m CosmoSim.dataset $< $@ 

$P-roulette.csv: $P-dataset.csv
	time python -m CosmoSim --toml $P-dataset.toml \
             --csvfile $< --outfile $@ \
             --directory images -vv  > $(log) 2> $(err)

$P-training.csv: $P-problem.toml $P-roulette.csv
	time python -m droulette.split $<

experiment%/model.pth: experiment%/ml.toml $P-training.csv
	time python -m droulette.model --config $< 

experiment%/eval.log: experiment%  experiment%/model.pt
	time python -m droulette.eval -o $</eval.csv $< > $@ 2> $</eval.err
	echo Done $<

clean:
	rm -f Dataset.ipynb
	cd experiment002 && $(MAKE) $@

distclean: clean
	rm -f *.csv
	rm -rf images resimulation
	rm -f experiment???/model.pth

ipynb: Dataset.ipynb
	cd experiment002 && $(MAKE) $@
J=jupytext --execute -w --to ipynb

%.ipynb: %.md
	$J "$<"

ipynb: Training.ipynb
	cd SIE ; $(MAKE) $@
	cd cluster2SIE ; $(MAKE) $@
	cd cluster4SIE ; $(MAKE) $@

J=jupytext --execute -w --to ipynb

%.ipynb: %.md
	$J "$<"

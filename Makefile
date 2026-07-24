ipynb: Training.ipynb
	cd SIE && $(MAKE) $@
	cd cluster2SIE && $(MAKE) $@
	cd cluster4SIS && $(MAKE) $@
clean:
	rm -f Training.ipynb
	cd SIE && $(MAKE) $@
	cd cluster2SIE && $(MAKE) $@
	cd cluster4SIS && $(MAKE) $@

J=jupytext --execute -w --to ipynb

%.ipynb: %.md
	$J "$<"

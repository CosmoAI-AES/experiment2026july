ipynb: Testing.ipynb Resimulation.ipynb 

J=jupytext --execute -w --to ipynb

%.ipynb: %.md
	$J "$<"

clean:
	rm -f *.ipynb

SIE-%.pdf: %.md
	myst build --pdf $<
pdf: SIE-Testing.pdf SIE-Resimulation.pdf

%.log: %.md
	$J "$<" > "$@" 2>&1 

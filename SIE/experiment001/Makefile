ipynb: Testing.ipynb Resimulation.ipynb 

J=jupytext --execute -w --to ipynb

%.ipynb: %.md
	$J "$<"

clean:
	rm -f *.ipynb

SIE-%-exec.pdf: %.md
	myst build --pdf --execute $<
SIE-%.pdf: %.md
	myst build --pdf $<
pdf: SIE-Testing.pdf SIE-Resimulation.pdf
exec: SIE-Testing-exec.pdf SIE-Resimulation-exec.pdf

%.log: %.md
	$J "$<" > "$@" 2>&1 

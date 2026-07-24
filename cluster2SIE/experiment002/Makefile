ipynb: Testing.ipynb 

J=jupytext --execute -w --to ipynb

%.ipynb: %.md
	$J "$<"

clean:
	rm -f *.ipynb

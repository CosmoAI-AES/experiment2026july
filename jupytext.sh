#!/bin/sh

find . -name Testing.md | xargs jupytext --sync --execute
find . -name Dataset.md | xargs jupytext --sync --execute
jupytext --sync --execute Training.md 

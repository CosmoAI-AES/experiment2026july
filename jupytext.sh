#!/bin/sh

find . -name Testing.md | xargs jupytext --sync --execute
jupytext --sync --execute Training.md SIE/Dataset.md

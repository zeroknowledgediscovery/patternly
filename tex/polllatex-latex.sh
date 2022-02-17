#!/bin/bash

texfile=$1
shift
inputs=$*

while true
do
#pdflatex -shell-escape $texfile.tex
latexmk -f -e '$pdflatex=q/pdflatex %O --shell-escape %S/' -pdf $texfile
sleep 2
done

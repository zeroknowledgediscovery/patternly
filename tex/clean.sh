#!/bin/bash

rm *fls
git rm *fls

git rm *~
git rm *out
git rm *log

rm *~
rm *out
rm *log

rm *aux*
git rm *aux*
rm *~
rm *out
rm *log
git rm *fls
git rm *aux
git rm *auxlock
git rm *latexmk

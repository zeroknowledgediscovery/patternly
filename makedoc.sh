#!/bin/bash

pdoc --skip-errors  --html patternly/ -o docs/ -c latex_math=True -f --template-dir docs/dark_templates

cp -r docs/patternly/* docs

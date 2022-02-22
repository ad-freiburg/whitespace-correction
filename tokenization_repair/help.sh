#!/bin/bash

echo "
      If you just want to look at the trained models and the demo:
        make demo (after executing go to your browser to the port you specified when starting this container)
      "

echo "
      If you want to reproduce the results by training yourself:
        Read the README.md for instructions (e.g. use 'cat README.md' or open it in the browser or a markdown viewer)
      "

echo "
      Additional development-related make targets (not relevant for the end user):
        make checkstyle (to check the code style using mypy and flake8)
        make tests      (to run unittests using pytest)
      "
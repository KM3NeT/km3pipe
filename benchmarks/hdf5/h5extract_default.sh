#!/bin/bash

for file in echo $(python -m km3net_testdata offline)/*.root; do
    echo "Extracting everything from $file"
    h5extract $file
    echo done
done


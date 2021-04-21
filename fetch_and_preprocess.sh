#!/bin/bash
set -e
# python2.7 scripts/download.py

CLASSPATH="lib:lib/stanford-parser/stanford-parser.jar:lib/stanford-parser/stanford-parser-3.5.1-models.jar"
javac -cp $CLASSPATH lib/*.java
python scripts/preprocess_xianer.py
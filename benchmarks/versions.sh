#!/bin/bash

XS_DIR=$HOME/code/cross-section
BENCHMARKS_DIR="$XS_DIR/benchmarks"
BIN_DIR=$BENCHMARKS_DIR/bin
LOGFILE=$BENCHMARKS_DIR/version_timings_current.log

mkdir $BIN_DIR
rm $BIN_DIR/*
rm $LOGFILE
touch $LOGFILE

for tag in $(git tag | sort); do
	echo "Compiling $tag"
	git checkout $tag
	binary="$BIN_DIR/perf_$tag"
	g++ -std=c++17 -O3 -I$XS_DIR -Wno-everything $BENCHMARKS_DIR/perf.cpp -o $binary

	runtime=$( /usr/bin/time -p "$binary" 2>&1 >/dev/null | grep real | awk '{print $2}' )
	echo "$tag: $runtime" >> $LOGFILE
done;

git checkout main

echo "done."
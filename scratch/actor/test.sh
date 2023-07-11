#!/bin/sh

awk '$1=="0" {print $2 "\t" $3 "\t" $4}' debug_cubic.txt >& debug_0.txt
awk '$1=="1" {print $2 "\t" $3 "\t" $4}' debug_cubic.txt >& debug_1.txt
awk '$1=="2" {print $2 "\t" $3 "\t" $4}' debug_cubic.txt >& debug_2.txt

awk '$2=="cwnd" {print $1 "\t" $3}' debug_0.txt >& debug_0_cwnd.txt
awk '$2=="cwnd" {print $1 "\t" $3}' debug_1.txt >& debug_1_cwnd.txt
awk '$2=="cwnd" {print $1 "\t" $3}' debug_2.txt >& debug_2_cwnd.txt
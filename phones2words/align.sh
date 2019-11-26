#!/bin/sh
#
# Align the phone output of image2speech into words, using a pronouncing lexicon
# Code originally by Justin van der hout, Zoltan D'Haese, and Justin van der Hout
# 11/26/2019
#
fstcompile --isymbols=vocab.letterlist --osymbols=vocab.letterlist example.fsm example.fst
fstcompose example.fst lexicon.fst output.fst
fstshortestpath output.fst output.shortest.fst 
fstproject --project_output output.shortest.fst | fstprint > output.txt

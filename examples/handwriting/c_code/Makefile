


all: Sim Sim-hdfjive

Sim: Handwriting.h LIFDesktop.c
	g++ -Wall -O2 -g LIFDesktop.c -o Sim

Output/flash.bin Handwriting.h: TextToBin
	./TextToBin > Handwriting.h
	@echo "New version of Handwriting.h has been generated"

TextToBin:
	gcc TextToBin.c -o TextToBin

.PHONY: clean

clean:
	rm -f Handwriting.h
	rm -f TextToBin
	rm -f Sim Sim-hdfjive
	rm -rf NetworkSetup/

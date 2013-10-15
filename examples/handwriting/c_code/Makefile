



Sim: Handwriting.h LIFDesktop.c
	g++ -Wall -O2 -g  -std=c99 LIFDesktop.c -o Sim

Output/flash.bin Handwriting.h: TextToBin
	./TextToBin > Handwriting.h
	@echo "New version of Handwriting.h has been generated"

TextToBin:
	gcc TextToBin.c -o TextToBin

.PHONY: clean

clean:
	rm -f Handwriting.h
	rm -f TextToBin
	rm -f Sim
	rm -rf NetworkSetup/

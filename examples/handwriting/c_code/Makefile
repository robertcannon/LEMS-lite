

all: Sim Sim-hdfjive

Sim: Handwriting.h LIFDesktop.c
	g++ -Wall -O2 -g  -std=c++0x LIFDesktop.c -o Sim

Sim-hdfjive: Handwriting.h LIFDesktop_visualise.cpp
	g++ -Wall -O2 -g  -std=c++0x LIFDesktop_visualise.cpp -o Sim-hdfjive -I"/auto/homes/mh735/hw/hdf-jive/include/" -L"/auto/homes/mh735/hw/hdf-jive/lib/" -lhdf5 -lhdf5_hl -lhdfjive
	echo 'export LD_LIBRARY_PATH="/auto/homes/mh735/hw/hdf-jive/lib/"'

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

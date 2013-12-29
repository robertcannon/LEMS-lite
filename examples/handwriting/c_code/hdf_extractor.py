import tables

f = tables.open_file("HandwritingResults.hdf")
#for node in f:
#    print node

#print f.root.Sim1.Pop1._f_getChild("0100").output_events.outspike.event_times.read()
#print f.root.Sim1.Pop1._f_getChild("0100").variables.v.raw.data.read()

spike_map = {}

population = f.root.Sim1.Pop1
for id, neuron in enumerate(population):
    spike_times = neuron.output_events.outspike.event_times.read()
    for time in spike_times:
        if time not in spike_map:
            spike_map[time] = [id]
        else:
            spike_map[time].append(id)


time_keys = spike_map.keys()
time_keys.sort()

fd = open("spikes.csv", "w")

print "\n\nTime Keys:\n" + str(time_keys)

for key in time_keys:
    spiking_neurons = spike_map[key]
    spiking_neurons.sort()

    spiking_neurons_string = "{0},{1}\n".format(int(key), ",".join(str(id) for id in spiking_neurons))
    print "\n\nSpiking Neurons:\n"
    print spiking_neurons_string + "-"*25 + "\n\n\n"

    fd.write(spiking_neurons_string)

fd.close()

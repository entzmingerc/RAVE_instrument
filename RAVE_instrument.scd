// shift + enter is run line
// ctrl  + enter is run block

// ctrl + M         is show VU meters
// ctrl + shift + M is show scope
// ctrl + alt   + T is show node tree

// OTHER miscellaneous checks
NN(\ravePerc).describe; // print stuff, ins and outs
NN(\ravePerc).methods[0]; // print stuff, methods available
NN(\raveEnsemble).methods[0].asString[29].asString.asInteger; // whacky way to print number of outputs
NN(\raveEnsemble).methods[0];
// model, prints the first method (forward), converts to string, indexes 29th char, char2string, string2int
NN.describeAll;
ServerOptions.outDevices; // print all devices, inDevices, outDevices
s.makeWindow; // record audio
"hello".postln; // test print

FreqScope.new(400, 200, 0, server: s);


s.reboot;
(
// microphone and soundin tutorial https://www.youtube.com/watch?v=3vu4UbS2NMw
s.options.safetyClipThreshold = 1;
s.options.inDevice = "Microphone (Scarlett 2i2 USB)";
s.options.outDevice = "Headphones (HyperX Cloud Alpha";
s.options.numOutputBusChannels = 4;
s.options.numInputBusChannels  = 2;
s.reboot;
s.meter; // VU meters corresponding to BusChannels
s.scope; // Oscilloscope corresponding to selected Audio or Control Rate Buses
)

// RAVE NN Initialization, Vintage, Percussion, NASA, VCTK models, each has a prior available
(
// https://forum.ircam.fr/projects/detail/rave/
NN.load(\ravePerc, "~/Documents/Recurse Center/pretrained models/percussion.ts");
NN.load(\raveVCTK, "~/Documents/Recurse Center/pretrained models/VCTK.ts");
NN.load(\raveNASA, "~/Documents/Recurse Center/pretrained models/nasa.ts");
NN.load(\raveVintage, "~/Documents/Recurse Center/pretrained models/vintage.ts");

// https://huggingface.co/Intelligent-Instruments-Lab/rave-models/tree/main
NN.load(\raveBirdsMother, "~/Documents/GitHub/RAVE-models/rave-models/birds_motherbird_b2048_r48000_z16.ts");
NN.load(\raveBirdsPluma, "~/Documents/GitHub/RAVE-models/rave-models/birds_pluma_b2048_r48000_z12.ts");
NN.load(\raveEnsemble, "~/Documents/GitHub/RAVE-models/rave-models/crozzoli_bigensemblesmusic_18d.ts");
NN.load(\raveGuitar, "~/Documents/GitHub/RAVE-models/rave-models/guitar_iil_b2048_r48000_z16.ts");
NN.load(\raveHumpbacks, "~/Documents/GitHub/RAVE-models/rave-models/humpbacks_pondbrain_b2048_r48000_z20.ts");
NN.load(\raveMagnets, "~/Documents/GitHub/RAVE-models/rave-models/magnets_b2048_r48000_z8.ts");
NN.load(\raveMarine, "~/Documents/GitHub/RAVE-models/rave-models/marinemammals_pondbrain_b2048_r48000_z20.ts");
NN.load(\raveResonator, "~/Documents/GitHub/RAVE-models/rave-models/mrp_strengjavera_b2048_r44100_z16.ts");
NN.load(\raveOrgan1, "~/Documents/GitHub/RAVE-models/rave-models/organ_archive_b2048_r48000_z16.ts");
NN.load(\raveOrgan2, "~/Documents/GitHub/RAVE-models/rave-models/organ_bach_b2048_r48000_z16.ts");
NN.load(\raveSaxophone, "~/Documents/GitHub/RAVE-models/rave-models/sax_soprano_franziskaschroeder_b2048_r48000_z20.ts");
NN.load(\raveVoice1, "~/Documents/GitHub/RAVE-models/rave-models/voice_hifitts_b2048_r48000_z16.ts");
NN.load(\raveVoice2, "~/Documents/GitHub/RAVE-models/rave-models/voice_jvs_b2048_r44100_z16.ts");
NN.load(\raveVoice3, "~/Documents/GitHub/RAVE-models/rave-models/voice_vctk_b2048_r44100_z22.ts");
NN.load(\raveVoice4, "~/Documents/GitHub/RAVE-models/rave-models/voice_vocalset_b2048_r48000_z16.ts");
NN.load(\raveVoice5, "~/Documents/GitHub/RAVE-models/rave-models/voice-multi-b2048-r48000-z11.ts");
NN.load(\raveWater, "~/Documents/GitHub/RAVE-models/rave-models/water_pondbrain_b2048_r48000_z16.ts");

)

// GENERIC SYNTHDEF FOR RAVE MODELS
var raveDict, raveDictBusIn, raveSynthDefs, raveNNModels, raveBusIn, nodeDict, freeOtherNodes, createNode;

(
~raveDict = Dictionary.new;
~raveDictBusIn = Dictionary.new;
~raveSynthDefs = [
	\NNPerc, \NNVCTK, \NNNASA, \NNVintage, \NNBirdsMother,
	\NNBirdsPluma,\NNEnsemble,\NNGuitar,\NNHumpbacks,\NNMagnets,
	\NNMarine,\NNResonator,\NNOrgan1,\NNOrgan2,\NNSaxophone,
	\NNVoice1,\NNVoice2,\NNVoice3, \NNVoice4, \NNVoice5,
	\NNWater]; // list of all rave SynthDefs
~raveNNModels = [
	\ravePerc, \raveVCTK, \raveNASA,  \raveVintage, \raveBirdsMother,
	\raveBirdsPluma,\raveEnsemble,\raveGuitar,\raveHumpbacks,\raveMagnets,
	\raveMarine,\raveResonator,\raveOrgan1,\raveOrgan2,\raveSaxophone,
	\raveVoice1,\raveVoice2,\raveVoice3, \raveVoice4, \raveVoice5,
	\raveWater]; // list of NN models loaded
~raveBusIn = [
	30,32,34,36,38,
	40,42,44,46,48,
	50,52,54,56,58,
	60,62,64,66,68,
	70];

// associate each SynthDef model with NN model loaded
for(0, ~raveSynthDefs.size - 1) { |i|
	i.postln;
	~raveSynthDefs[i].postln;
	~raveNNModels[i].postln;
	~raveDict.put(~raveSynthDefs[i], ~raveNNModels[i]);
	~raveDictBusIn.put(~raveSynthDefs[i],~raveBusIn[i]);
};
~raveDict.getPairs.postln;
~raveDictBusIn.getPairs.postln;
~raveSynthDefs.postln;


// SYNTH DEFS
~raveSynthDefs.do{|name|
	var synthdefname = "r" ++ name.asString;
	synthdefname.postln;

	SynthDef.new(synthdefname, { |sig, in, out=0, latent, modLatent, prior, num_outs|
		in = ~raveDictBusIn[name];
		latent = NN(~raveDict[name], \encode).ar(In.ar(in, 1));
		modLatent = latent.collect { |l|
			l + LFNoise1.ar(MouseY.kr.exprange(0.1, 30)).range(-0.5, 0.5)
		};
		sig = NN(~raveDict[name], \decode).ar(modLatent);
		sig = sig * 0.85;
		num_outs = NN(~raveDict[name]).methods[0].asString[29].asString.asInteger; // whacky way to get number of outputs
		if (num_outs < 2, {Out.ar(out, sig); Out.ar(out+1, sig);}, {Out.ar(out, sig);})
		// Out.ar(out, sig); // idk figure this out next based on the
    }).add;
};


~nodeDict = Dictionary.new;

~freeOtherNodes = {
	~nodeDict.keysValuesDo { |key, value|
		if (key != "audioInSend") {
			value.node.free;
			value.watcher.free;
			~nodeDict.removeAt(key);
		}
	}
};

~createNode = { |label|
	var newNode, newWatcher, labelExists, nodeLocation;
	labelExists = ~nodeDict.includesKey(label);
	if(labelExists, {
		("deleting " ++ label).postln;
		newNode = ~nodeDict[label];
		newNode.node.free;
		newNode.watcher.free;
		~nodeDict.removeAt(label);
	}, {
		("creating " ++ label).postln;
		label.postln;
		if ((label == "audioInSend").or(label == "audioSend"), {
			nodeLocation = 'addToHead';
		}, {
			nodeLocation = 'addToTail';});
		nodeLocation.postln;
		newNode = Synth.new(label, [], 1, nodeLocation);
		newWatcher = NodeWatcher.register(newNode);
		~nodeDict[label] = (node: newNode, watcher: newWatcher);
	});
};

// AUDIO SYNTH DEFS
SynthDef("audioSend", {
	Out.ar([
	30,32,34,36,38,
	40,42,44,46,48,
	50,52,54,56,58,
	60,62,64,66,68,
	70], SoundIn.ar(0, 1)); // send to RAVE NN
}).add;

SynthDef("audioInSend", {
	Out.ar([0, 1,
	30,32,34,36,38,
	40,42,44,46,48,
	50,52,54,56,58,
	60,62,64,66,68,
	70], SoundIn.ar(0, 1)); // send to output LR and send to RAVE NN
}).add;

SynthDef("audioIn", {
	Out.ar([0, 1], SoundIn.ar(0, 1)); // send to output LR
}).add;

"FINISHED FUNCTION INIT".postln;
)

/*
IRCAM
https://forum.ircam.fr/projects/detail/rave/
Intelligent Instruments Labs
https://huggingface.co/Intelligent-Instruments-Lab/rave-models/tree/main
*/

// AUDIO INPUT
~createNode.value("audioInSend"); // only one pls
~createNode.value("audioSend"); // only one pls

// RAVE models: real-time machine learning timbre transformation
~createNode.value("rNNPerc");   // percussion
~createNode.value("rNNGuitar"); // "guitar"
~createNode.value("rNNMagnets"); // physical metallic scrapes
~createNode.value("rNNWater");  // bathroomcore
~createNode.value("rNNVintage"); // fallout old timey music
~createNode.value("rNNHumpbacks"); // whales??
~createNode.value("rNNMarine"); // ...not whales??
~createNode.value("rNNOrgan1"); // worn tape kinda noisy
~createNode.value("rNNOrgan2"); // pretty creepy
~createNode.value("rNNNASA");   // harsh noise with high volume
~createNode.value("rNNVCTK");   // vocal
~createNode.value("rNNVoice1"); // vocal
~createNode.value("rNNVoice3"); // vocal
~createNode.value("rNNVoice4"); // vocal
~createNode.value("rNNVoice5"); // vocal

~freeOtherNodes.value; // clear RAVEs
s.freeAll; // 10 at a time is limit



// GRAVEYARD
// Synth.new("rNNSaxophone"); // doesn't seem to work
// Synth.new("rNNVoice2"); // isn't trained on silence :(
// Synth.new("rNNResonator"); // doesn't track pitch
// Synth.new("rNNEnsemble"); // uhhh 18 outputs? o_O? sounds cool tho
// Synth.new("rNNBirdsMother"); // high hats make it harsh noise
// Synth.new("rNNBirdsPluma"); // that's not a bird, ouch my ears
// audioIn, audioInSend, audioSend == dry/wet mix 0% to 50/50 to 100% rave
// voice model works best with tonal noises, techno drums don't do much
// also lots of high frequency artifacts when pushing the input when clipping

/*~nodeDict.getPairs.postln;
~nodeDict["rNNPerc"].postln;
~nodeDict.removeAt("rNNPerc");
var node1, node2, watcher1, watcher2;
(
~node1 = Synth(\audioInSend);
~watcher1 = NodeWatcher.register(~node1);
)

(
~node2 = Synth("rNNWater", [], 1, 'addToTail');
~watcher2 = NodeWatcher.register(~node2);
)
~node1.free;
~node2.free;
s.freeAll;
// nodeWatcher
(
Synth.new(\audioIn);
)*/

/*(
// Synth.new("rNNPerc"); // kinda loud output, high quality
// Synth.new("rNNVCTK");
// Synth.new("rNNNASA"); // harsh noise with high vol input
// Synth.new("rNNVintage");

// Synth.new("rNNGuitar");
// Synth.new("rNNHumpbacks");
// Synth.new("rNNMagnets"); // physical metallic scrapes
// Synth.new("rNNMarine");
// Synth.new("rNNOrgan1"); // worn tape kinda noisy
// Synth.new("rNNOrgan2"); // pretty creepy
// Synth.new("rNNVoice1");
// Synth.new("rNNVoice3");
// Synth.new("rNNVoice4");
// Synth.new("rNNVoice5");
Synth.new("rNNWater");
Synth.new(\audioInSend); // 50/50 dry/send to RAVE
// Synth.new(\audioSend); // send all to RAVE

)
s.freeAll;*/
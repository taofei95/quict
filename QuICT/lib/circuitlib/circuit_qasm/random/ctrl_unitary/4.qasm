OPENQASM 2.0;
include "qelib1.inc";
qreg q[4];
creg c[4];
cx q[1], q[0];
cx q[2], q[3];
cx q[3], q[0];
cx q[1], q[0];
ch q[3], q[0];
ch q[2], q[3];
cy q[0], q[3];
cu3(3.185976297329146, 2.649556718361627, 1.2403018976224238) q[3], q[2];
cy q[0], q[2];
cy q[1], q[0];
cx q[1], q[3];
ch q[2], q[1];
cu3(3.851885792186718, 0.8051407171083478, 2.3518915851456503) q[1], q[3];
ch q[2], q[3];
cy q[3], q[0];
cu3(5.9402153692461415, 0.3137834341860858, 2.5711513982694942) q[3], q[1];
ch q[3], q[0];
ch q[1], q[2];
cx q[1], q[2];
cu3(5.494800986172512, 6.121372725336458, 2.98467011040129) q[3], q[2];
ch q[3], q[2];
cx q[1], q[0];
cu3(5.504439603114704, 5.978509121826732, 1.259940234630211) q[0], q[2];
cx q[1], q[2];
cx q[3], q[0];
cu3(3.287187689610157, 3.2641892260252323, 1.8141216697671352) q[0], q[3];
cx q[2], q[3];
ch q[3], q[0];
ch q[0], q[2];
ch q[3], q[2];
cy q[0], q[3];
ch q[2], q[3];
cy q[1], q[2];
cy q[3], q[0];
cu3(2.3268951899446018, 4.10610378465781, 0.4572769910815831) q[1], q[3];
cy q[0], q[2];
cu3(1.4735082926484244, 0.3632577784507847, 4.4933317851354975) q[1], q[3];
ch q[0], q[2];
cy q[2], q[1];
cu3(4.933387312325908, 4.570693861213375, 2.3377884516818668) q[0], q[3];
cx q[2], q[0];
cx q[0], q[2];
cu3(3.4466271177085397, 5.617380043986046, 4.303275528655404) q[3], q[1];
cy q[0], q[1];
cy q[2], q[0];
ch q[1], q[2];
ch q[0], q[1];
ch q[1], q[2];
ch q[2], q[3];
cy q[2], q[1];
cy q[0], q[3];
cx q[2], q[3];
ch q[0], q[2];
cu3(5.7173055715366825, 0.9295842351351166, 3.45222035447406) q[0], q[3];
cu3(1.1206755391830079, 6.071645450176169, 1.7850657577715134) q[3], q[2];
cx q[1], q[2];
cu3(5.84333216034413, 2.740963048968077, 3.726469886141606) q[0], q[2];
cu3(3.5499319998982295, 0.350874024169067, 1.7838848187792857) q[2], q[3];
ch q[3], q[0];
cu3(4.818273006147838, 1.7234901654272847, 0.2972868395372693) q[2], q[3];
cx q[3], q[0];
cx q[1], q[2];
cx q[0], q[3];
cy q[2], q[1];
cx q[2], q[0];
ch q[0], q[2];
ch q[0], q[3];
cu3(5.488406593136821, 0.8948452256907974, 3.7901151055200217) q[3], q[2];
cu3(0.04174118085067542, 1.4928021893961754, 0.43949510115031154) q[3], q[1];
cx q[0], q[2];
cx q[0], q[2];
cu3(0.31924616189882854, 2.819760464107434, 3.2610225892357434) q[2], q[3];
cy q[3], q[1];
ch q[2], q[1];
ch q[1], q[2];
ch q[0], q[1];
cu3(0.5476537916812031, 0.7514649658461123, 2.2969357490558178) q[2], q[3];
cy q[3], q[1];
cu3(0.07726342600251308, 2.1406556544516326, 4.952461419752637) q[2], q[0];
cu3(2.902015673368705, 0.5499772002866177, 0.021097898588901577) q[0], q[3];
ch q[3], q[1];
cy q[1], q[3];
ch q[1], q[2];
cu3(4.997818514227958, 0.0735688173387454, 3.5876525244789814) q[3], q[0];
cx q[2], q[3];
cu3(3.2141845427319007, 3.589907357953226, 0.580546357065461) q[0], q[3];
cu3(2.2319723893681167, 1.1045751473161796, 1.5919639007159683) q[1], q[3];
cx q[1], q[3];
ch q[2], q[0];
ch q[0], q[1];
cx q[2], q[3];
cy q[0], q[3];
cy q[2], q[3];
cu3(0.04324480297615703, 1.5544281452669955, 5.035588195895044) q[1], q[3];
ch q[3], q[0];
cu3(5.78584836952611, 4.672044385646906, 5.065036516994762) q[2], q[3];
cy q[2], q[0];
ch q[1], q[2];
cy q[0], q[3];
cu3(4.153551208828279, 1.7998517428994698, 3.8441228637377063) q[2], q[0];
ch q[2], q[3];
cx q[0], q[2];
cu3(0.40681875587370275, 3.613004962618937, 2.649990120978165) q[0], q[2];
ch q[1], q[0];
cu3(1.3730562998112734, 3.5195883233683873, 1.09515102741821) q[2], q[3];
ch q[2], q[0];
cu3(3.4496864184783465, 4.1749724658912655, 2.7985224373738617) q[3], q[2];
ch q[1], q[3];
cx q[0], q[3];
cx q[1], q[2];
cu3(0.577274186580312, 2.1888661071656426, 1.0611604371363188) q[3], q[0];
cu3(2.8445257814281595, 3.159214752606411, 6.126996366546351) q[1], q[2];
cx q[0], q[3];
ch q[2], q[0];
cu3(5.272202073358632, 4.834919590244037, 2.3716779980190195) q[0], q[3];
cy q[1], q[2];
cy q[2], q[1];
ch q[1], q[0];
cy q[3], q[1];
cu3(0.8967128105129023, 1.5518115167755246, 3.016045417102714) q[1], q[3];
ch q[3], q[0];
cu3(4.675280756058939, 0.10899276929594842, 2.3404308993670013) q[1], q[3];
cx q[2], q[0];
cy q[1], q[3];
cu3(5.8836151994440735, 3.045543648004763, 3.8731470997106014) q[1], q[0];
cx q[2], q[0];
ch q[0], q[3];
cu3(1.4303509653996584, 5.751271496051235, 6.024590351629065) q[3], q[0];
cx q[0], q[1];
cx q[3], q[2];
ch q[2], q[0];
cu3(0.08930019173858318, 1.0801960208824535, 3.1669083234594106) q[1], q[3];
cx q[2], q[1];
cx q[1], q[2];
cx q[1], q[3];
cx q[3], q[1];
cx q[3], q[1];
cx q[1], q[3];
cu3(2.7262664238455305, 5.3029772885721576, 4.51351783379197) q[3], q[2];
cu3(1.8698708375302853, 2.0152151010019885, 3.049053901447609) q[0], q[1];
cx q[0], q[1];
cx q[1], q[0];
cx q[0], q[1];
cu3(4.19973834164815, 0.4789575052206889, 4.467230356466038) q[0], q[2];
ch q[0], q[1];
cx q[3], q[2];
cy q[1], q[2];
ch q[1], q[2];
cx q[1], q[0];
cu3(2.010194111657726, 1.9430139766139882, 4.729955442504447) q[0], q[1];
cy q[0], q[1];
cy q[2], q[1];
cx q[1], q[3];
cx q[2], q[3];
cu3(4.463654666371486, 4.286371309544355, 3.0544706362180114) q[3], q[2];
cy q[2], q[3];
ch q[1], q[0];
cy q[3], q[1];
cu3(4.51070772566786, 0.7055459657685305, 4.112219623014973) q[3], q[2];
cy q[0], q[3];
OPENQASM 2.0;
include "qelib1.inc";
qreg q[73];
creg c[73];
x q[0];
x q[4];
x q[5];
x q[7];
x q[8];
x q[10];
x q[11];
x q[12];
x q[13];
x q[15];
x q[19];
x q[23];
x q[24];
x q[25];
x q[27];
x q[28];
x q[29];
x q[33];
x q[34];
x q[36];
x q[38];
x q[43];
x q[44];
x q[45];
x q[46];
x q[48];
x q[49];
x q[51];
x q[55];
x q[56];
x q[60];
x q[66];
x q[68];
x q[70];
x q[71];
x q[0];
h q[0];
rzz(0.3671663999557495) q[0], q[72];
rzz(0.9168698191642761) q[1], q[72];
rzz(0.34013891220092773) q[2], q[72];
rzz(0.03392612934112549) q[3], q[72];
rzz(0.24897778034210205) q[4], q[72];
rzz(0.001012563705444336) q[5], q[72];
rzz(0.9724072217941284) q[6], q[72];
rzz(0.5287243723869324) q[7], q[72];
rzz(0.3631448745727539) q[8], q[72];
rzz(0.1266622543334961) q[9], q[72];
rzz(0.8027777075767517) q[10], q[72];
rzz(0.2421281933784485) q[11], q[72];
rzz(0.15852141380310059) q[12], q[72];
rzz(0.45158106088638306) q[13], q[72];
rzz(0.9869694709777832) q[14], q[72];
rzz(0.8531332015991211) q[15], q[72];
rzz(0.6079156398773193) q[16], q[72];
rzz(0.31669092178344727) q[17], q[72];
rzz(0.7302589416503906) q[18], q[72];
rzz(0.375948429107666) q[19], q[72];
rzz(0.20006638765335083) q[20], q[72];
rzz(0.7285189628601074) q[21], q[72];
rzz(0.459786593914032) q[22], q[72];
rzz(0.09979885816574097) q[23], q[72];
rzz(0.947736918926239) q[24], q[72];
rzz(0.8644664883613586) q[25], q[72];
rzz(0.04410755634307861) q[26], q[72];
rzz(0.6801337599754333) q[27], q[72];
rzz(0.466660737991333) q[28], q[72];
rzz(0.6495742201805115) q[29], q[72];
rzz(0.5163880586624146) q[30], q[72];
rzz(0.8304954767227173) q[31], q[72];
rzz(0.05983865261077881) q[32], q[72];
rzz(0.6800378561019897) q[33], q[72];
rzz(0.7745634317398071) q[34], q[72];
rzz(0.29019486904144287) q[35], q[72];
rzz(0.1669667363166809) q[36], q[72];
rzz(0.36777108907699585) q[37], q[72];
rzz(0.7174480557441711) q[38], q[72];
rzz(0.20449531078338623) q[39], q[72];
rzz(0.7351207137107849) q[40], q[72];
rzz(0.8575116395950317) q[41], q[72];
rzz(0.9910063147544861) q[42], q[72];
rzz(0.9630452394485474) q[43], q[72];
rzz(0.7057867050170898) q[44], q[72];
rzz(0.694462776184082) q[45], q[72];
rzz(0.9242433905601501) q[46], q[72];
rzz(0.6296166181564331) q[47], q[72];
rzz(0.42868155241012573) q[48], q[72];
rzz(0.6008450984954834) q[49], q[72];
rzz(0.5005207061767578) q[50], q[72];
rzz(0.18627554178237915) q[51], q[72];
rzz(0.6857169270515442) q[52], q[72];
rzz(0.7318604588508606) q[53], q[72];
rzz(0.9390533566474915) q[54], q[72];
rzz(0.9362601041793823) q[55], q[72];
rzz(0.14908844232559204) q[56], q[72];
rzz(0.7442426085472107) q[57], q[72];
rzz(0.23698967695236206) q[58], q[72];
rzz(0.7220969200134277) q[59], q[72];
rzz(0.7300967574119568) q[60], q[72];
rzz(0.8032889366149902) q[61], q[72];
rzz(0.12798666954040527) q[62], q[72];
rzz(0.5754607319831848) q[63], q[72];
rzz(0.06437540054321289) q[64], q[72];
rzz(0.5108321309089661) q[65], q[72];
rzz(0.13035821914672852) q[66], q[72];
rzz(0.6662914752960205) q[67], q[72];
rzz(0.18647992610931396) q[68], q[72];
rzz(0.6940943598747253) q[69], q[72];
rzz(0.6064870953559875) q[70], q[72];
rzz(0.8324349522590637) q[71], q[72];
rzz(0.6982457637786865) q[0], q[72];
rzz(0.17652475833892822) q[1], q[72];
rzz(0.07472068071365356) q[2], q[72];
rzz(0.33102595806121826) q[3], q[72];
rzz(0.7778925895690918) q[4], q[72];
rzz(0.9619291424751282) q[5], q[72];
rzz(0.6200941205024719) q[6], q[72];
rzz(0.42847275733947754) q[7], q[72];
rzz(0.48648256063461304) q[8], q[72];
rzz(0.4112600088119507) q[9], q[72];
rzz(0.8574104309082031) q[10], q[72];
rzz(0.30389875173568726) q[11], q[72];
rzz(0.836337685585022) q[12], q[72];
rzz(0.15242773294448853) q[13], q[72];
rzz(0.6026989817619324) q[14], q[72];
rzz(0.7704719305038452) q[15], q[72];
rzz(0.21951574087142944) q[16], q[72];
rzz(0.8026701807975769) q[17], q[72];
rzz(0.5910052061080933) q[18], q[72];
rzz(0.6916492581367493) q[19], q[72];
rzz(0.8103996515274048) q[20], q[72];
rzz(0.2136327624320984) q[21], q[72];
rzz(0.4623498320579529) q[22], q[72];
rzz(0.8610817790031433) q[23], q[72];
rzz(0.38261789083480835) q[24], q[72];
rzz(0.699576735496521) q[25], q[72];
rzz(0.8433999419212341) q[26], q[72];
rzz(0.3226191997528076) q[27], q[72];
rzz(0.9450237154960632) q[28], q[72];
rzz(0.714225709438324) q[29], q[72];
rzz(0.13537442684173584) q[30], q[72];
rzz(0.8574594855308533) q[31], q[72];
rzz(0.0005986690521240234) q[32], q[72];
rzz(0.47858959436416626) q[33], q[72];
rzz(0.1399046778678894) q[34], q[72];
rzz(0.42051881551742554) q[35], q[72];
rzz(0.03029930591583252) q[36], q[72];
rzz(0.08325964212417603) q[37], q[72];
rzz(0.9035735726356506) q[38], q[72];
rzz(0.39734941720962524) q[39], q[72];
rzz(0.47708767652511597) q[40], q[72];
rzz(0.6376965045928955) q[41], q[72];
rzz(0.024547815322875977) q[42], q[72];
rzz(0.8665042519569397) q[43], q[72];
rzz(0.8504265546798706) q[44], q[72];
rzz(0.6817410588264465) q[45], q[72];
rzz(0.415860116481781) q[46], q[72];
rzz(0.5388047099113464) q[47], q[72];
rzz(0.6238601803779602) q[48], q[72];
rzz(0.7545296549797058) q[49], q[72];
rzz(0.8221745491027832) q[50], q[72];
rzz(0.24754077196121216) q[51], q[72];
rzz(0.027875781059265137) q[52], q[72];
rzz(0.10760748386383057) q[53], q[72];
rzz(0.9792315363883972) q[54], q[72];
rzz(0.032518863677978516) q[55], q[72];
rzz(0.3274526596069336) q[56], q[72];
rzz(0.9822667837142944) q[57], q[72];
rzz(0.9881035685539246) q[58], q[72];
rzz(0.9615733027458191) q[59], q[72];
rzz(0.8110604882240295) q[60], q[72];
rzz(0.004465997219085693) q[61], q[72];
rzz(0.07363039255142212) q[62], q[72];
rzz(0.07863259315490723) q[63], q[72];
rzz(0.5098105072975159) q[64], q[72];
rzz(0.8524113893508911) q[65], q[72];
rzz(0.9745215177536011) q[66], q[72];
rzz(0.9255390763282776) q[67], q[72];
rzz(0.8543584942817688) q[68], q[72];
rzz(0.6782748103141785) q[69], q[72];
rzz(0.048224568367004395) q[70], q[72];
rzz(0.19642764329910278) q[71], q[72];
h q[0];

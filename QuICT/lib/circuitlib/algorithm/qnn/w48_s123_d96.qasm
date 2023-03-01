OPENQASM 2.0;
include "qelib1.inc";
qreg q[48];
creg c[48];
x q[2];
x q[3];
x q[5];
x q[7];
x q[8];
x q[9];
x q[10];
x q[11];
x q[12];
x q[13];
x q[16];
x q[17];
x q[19];
x q[20];
x q[22];
x q[23];
x q[24];
x q[27];
x q[28];
x q[31];
x q[35];
x q[37];
x q[38];
x q[42];
x q[44];
x q[46];
x q[0];
h q[0];
rzz(0.28683382272720337) q[0], q[47];
rzz(0.8162059783935547) q[1], q[47];
rzz(0.08096200227737427) q[2], q[47];
rzz(0.7042184472084045) q[3], q[47];
rzz(0.974065363407135) q[4], q[47];
rzz(0.908672571182251) q[5], q[47];
rzz(0.6212465763092041) q[6], q[47];
rzz(0.9224785566329956) q[7], q[47];
rzz(0.7039047479629517) q[8], q[47];
rzz(0.7073902487754822) q[9], q[47];
rzz(0.9977070689201355) q[10], q[47];
rzz(0.5979366898536682) q[11], q[47];
rzz(0.4918420910835266) q[12], q[47];
rzz(0.01139754056930542) q[13], q[47];
rzz(0.8558220267295837) q[14], q[47];
rzz(0.10289067029953003) q[15], q[47];
rzz(0.27490168809890747) q[16], q[47];
rzz(0.35180026292800903) q[17], q[47];
rzz(0.6308963894844055) q[18], q[47];
rzz(0.5122370719909668) q[19], q[47];
rzz(0.15334433317184448) q[20], q[47];
rzz(0.23276013135910034) q[21], q[47];
rzz(0.589764416217804) q[22], q[47];
rzz(0.23761385679244995) q[23], q[47];
rzz(0.39872992038726807) q[24], q[47];
rzz(0.546798825263977) q[25], q[47];
rzz(0.036296546459198) q[26], q[47];
rzz(0.6568868160247803) q[27], q[47];
rzz(0.5848331451416016) q[28], q[47];
rzz(0.27787381410598755) q[29], q[47];
rzz(0.282711923122406) q[30], q[47];
rzz(0.606705367565155) q[31], q[47];
rzz(0.8788121938705444) q[32], q[47];
rzz(0.7002520561218262) q[33], q[47];
rzz(0.39598995447158813) q[34], q[47];
rzz(0.04775643348693848) q[35], q[47];
rzz(0.39386874437332153) q[36], q[47];
rzz(0.45954304933547974) q[37], q[47];
rzz(0.6763184070587158) q[38], q[47];
rzz(0.12418866157531738) q[39], q[47];
rzz(0.5788552761077881) q[40], q[47];
rzz(0.2997153401374817) q[41], q[47];
rzz(0.6821714639663696) q[42], q[47];
rzz(0.6768513321876526) q[43], q[47];
rzz(0.7850440740585327) q[44], q[47];
rzz(0.7960073351860046) q[45], q[47];
rzz(0.4729887843132019) q[46], q[47];
rzz(0.4361029863357544) q[0], q[47];
rzz(0.8439419865608215) q[1], q[47];
rzz(0.5806962251663208) q[2], q[47];
rzz(0.6293942928314209) q[3], q[47];
rzz(0.7042417526245117) q[4], q[47];
rzz(0.6346743702888489) q[5], q[47];
rzz(0.5211203098297119) q[6], q[47];
rzz(0.8700798749923706) q[7], q[47];
rzz(0.4322606325149536) q[8], q[47];
rzz(0.25474339723587036) q[9], q[47];
rzz(0.5629592537879944) q[10], q[47];
rzz(0.8386030197143555) q[11], q[47];
rzz(0.9414478540420532) q[12], q[47];
rzz(0.7289822697639465) q[13], q[47];
rzz(0.9251287579536438) q[14], q[47];
rzz(0.20271837711334229) q[15], q[47];
rzz(0.2538022994995117) q[16], q[47];
rzz(0.5985907912254333) q[17], q[47];
rzz(0.42991721630096436) q[18], q[47];
rzz(0.3922368288040161) q[19], q[47];
rzz(0.2240191102027893) q[20], q[47];
rzz(0.16832327842712402) q[21], q[47];
rzz(0.6023454070091248) q[22], q[47];
rzz(0.8292326927185059) q[23], q[47];
rzz(0.6436208486557007) q[24], q[47];
rzz(0.4672689437866211) q[25], q[47];
rzz(0.5505849123001099) q[26], q[47];
rzz(0.7407189011573792) q[27], q[47];
rzz(0.08876574039459229) q[28], q[47];
rzz(0.8062823414802551) q[29], q[47];
rzz(0.981592059135437) q[30], q[47];
rzz(0.37637168169021606) q[31], q[47];
rzz(0.869661271572113) q[32], q[47];
rzz(0.18622463941574097) q[33], q[47];
rzz(0.9294009208679199) q[34], q[47];
rzz(0.160805344581604) q[35], q[47];
rzz(0.1265031099319458) q[36], q[47];
rzz(0.9382238388061523) q[37], q[47];
rzz(0.16252195835113525) q[38], q[47];
rzz(0.18651562929153442) q[39], q[47];
rzz(0.6672883629798889) q[40], q[47];
rzz(0.6011092662811279) q[41], q[47];
rzz(0.23489147424697876) q[42], q[47];
rzz(0.11203700304031372) q[43], q[47];
rzz(0.5866153836250305) q[44], q[47];
rzz(0.8580775856971741) q[45], q[47];
rzz(0.5481635332107544) q[46], q[47];
h q[0];
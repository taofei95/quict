OPENQASM 2.0;
include "qelib1.inc";
qreg q[19];
creg c[19];
x q[2];
x q[9];
x q[10];
x q[13];
x q[14];
x q[15];
x q[17];
x q[0];
h q[0];
rzz(0.9921954870223999) q[0], q[18];
rzz(0.9666396379470825) q[1], q[18];
rzz(0.8409408330917358) q[2], q[18];
rzz(0.8866385817527771) q[3], q[18];
rzz(0.8536199927330017) q[4], q[18];
rzz(0.9343044757843018) q[5], q[18];
rzz(0.47521501779556274) q[6], q[18];
rzz(0.6904521584510803) q[7], q[18];
rzz(0.7156892418861389) q[8], q[18];
rzz(0.10526734590530396) q[9], q[18];
rzz(0.40769386291503906) q[10], q[18];
rzz(0.720103919506073) q[11], q[18];
rzz(0.6774179935455322) q[12], q[18];
rzz(0.8934714198112488) q[13], q[18];
rzz(0.3273460865020752) q[14], q[18];
rzz(0.8396019339561462) q[15], q[18];
rzz(0.2871379256248474) q[16], q[18];
rzz(0.7513534426689148) q[17], q[18];
rzz(0.8226402997970581) q[0], q[18];
rzz(0.35542482137680054) q[1], q[18];
rzz(0.6394932866096497) q[2], q[18];
rzz(0.5661149024963379) q[3], q[18];
rzz(0.027010083198547363) q[4], q[18];
rzz(0.881256103515625) q[5], q[18];
rzz(0.6843189001083374) q[6], q[18];
rzz(0.1325894594192505) q[7], q[18];
rzz(0.8574108481407166) q[8], q[18];
rzz(0.6709892153739929) q[9], q[18];
rzz(0.3020473122596741) q[10], q[18];
rzz(0.8000050187110901) q[11], q[18];
rzz(0.44893962144851685) q[12], q[18];
rzz(0.3845871090888977) q[13], q[18];
rzz(0.5901346206665039) q[14], q[18];
rzz(0.5456656217575073) q[15], q[18];
rzz(0.05533707141876221) q[16], q[18];
rzz(0.7881034016609192) q[17], q[18];
h q[0];

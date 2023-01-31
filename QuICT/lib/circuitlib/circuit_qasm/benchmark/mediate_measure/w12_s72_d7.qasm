OPENQASM 2.0;
include "qelib1.inc";
qreg q[12];
creg c[12];
rz(0.024284438288841897) q[6];
rz(4.7558648411769235) q[4];
rz(1.6739847897403317) q[7];
rz(3.7287037112105095) q[8];
rz(6.182237295474987) q[9];
rz(3.1130296214924127) q[0];
rz(4.34412300757333) q[1];
rz(2.5466572776173244) q[11];
rz(5.036369529258013) q[10];
rz(3.079969710247782) q[2];
rz(5.7973900579477196) q[5];
rz(2.349151151885466) q[3];
cx q[1], q[2];
rz(1.688853046789658) q[11];
rz(6.000429050796255) q[4];
rz(2.6444431062679543) q[6];
rz(4.642010689270899) q[5];
rz(0.6412141660678288) q[3];
rz(5.382181843702617) q[9];
rz(1.062020090658221) q[0];
cx q[10], q[8];
rz(4.78563975373198) q[7];
rz(3.3313831382031145) q[10];
rz(4.224994794037125) q[4];
rz(2.328721362179964) q[1];
cx q[7], q[3];
rz(3.402374255131722) q[11];
rz(3.6139209703639876) q[9];
rz(1.038833398027411) q[8];
rz(0.03777209906784758) q[0];
rz(0.7508820325945105) q[6];
rz(4.327613444813347) q[2];
rz(6.197987533105292) q[5];
rz(1.6879281602251437) q[5];
rz(5.781787110846666) q[9];
rz(2.924756493970758) q[4];
measure q[0] -> c[0];
measure q[1] -> c[1];
measure q[2] -> c[2];
measure q[3] -> c[3];
measure q[4] -> c[4];
measure q[5] -> c[5];
measure q[6] -> c[6];
measure q[7] -> c[7];
measure q[8] -> c[8];
measure q[9] -> c[9];
measure q[10] -> c[10];
measure q[11] -> c[11];
rz(1.4146021268877325) q[2];
rz(5.851819862485422) q[11];
rz(1.729279642055497) q[7];
cx q[1], q[3];
rz(2.583342461297966) q[10];
cx q[8], q[0];
rz(4.068429929419112) q[6];
rz(1.3558038876366503) q[8];
rz(3.9511570073078923) q[3];
cx q[1], q[5];
rz(3.2818042262775204) q[4];
cx q[2], q[6];
cx q[7], q[11];
rz(1.5648130020066888) q[9];
rz(2.18854846708762) q[10];
rz(5.986279877542743) q[0];
rz(2.7662674422569054) q[3];
rz(0.5832723279001967) q[5];
rz(5.496816319515558) q[1];
rz(5.252562237273704) q[11];
rz(4.527711402999194) q[9];
rz(1.6366760422671505) q[8];
rz(4.764140319547115) q[2];
cx q[4], q[7];
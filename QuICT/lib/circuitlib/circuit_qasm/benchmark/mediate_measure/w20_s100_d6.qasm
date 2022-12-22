OPENQASM 2.0;
include "qelib1.inc";
qreg q[20];
creg c[20];
rz(5.86139038260558) q[8];
rz(0.7464128248005324) q[2];
rz(5.028675585499151) q[13];
rz(0.7023599346936694) q[19];
rz(4.91170376134442) q[12];
rz(4.256268776839054) q[9];
cx q[15], q[18];
cx q[5], q[1];
rz(5.626489477201228) q[10];
rz(0.5110599413512087) q[14];
rz(0.5400682728785521) q[0];
rz(3.4803964925629884) q[7];
rz(0.9607302941347643) q[17];
rz(5.516977275060617) q[4];
rz(2.220200035914365) q[16];
cx q[3], q[6];
rz(1.459742530135795) q[11];
rz(3.8169071570390347) q[7];
rz(0.7996839793471021) q[14];
rz(1.05737066978905) q[10];
rz(3.3028380152933092) q[5];
rz(4.96898758221697) q[3];
rz(4.881143808015504) q[12];
rz(1.7843686794562623) q[18];
rz(1.8009837978039442) q[16];
rz(2.862571244972978) q[11];
cx q[19], q[13];
rz(6.169026262992023) q[2];
cx q[17], q[1];
rz(3.7919997310346765) q[0];
rz(0.6352478312704034) q[15];
cx q[8], q[9];
rz(3.662419977878678) q[6];
rz(3.2316816850973877) q[4];
rz(5.356808531518436) q[14];
rz(5.54388858724054) q[13];
cx q[2], q[11];
rz(5.120655008184313) q[9];
rz(4.640423801391167) q[5];
rz(1.8997517311859693) q[19];
rz(0.19818564206466474) q[10];
rz(5.170274560090369) q[18];
cx q[3], q[7];
rz(5.576575883769701) q[6];
rz(1.9504824664325555) q[16];
rz(1.8929148497813246) q[0];
rz(5.068440733500964) q[17];
cx q[4], q[1];
rz(4.131431068651268) q[12];
rz(4.814612825358509) q[8];
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
measure q[12] -> c[12];
measure q[13] -> c[13];
measure q[14] -> c[14];
measure q[15] -> c[15];
measure q[16] -> c[16];
measure q[17] -> c[17];
measure q[18] -> c[18];
measure q[19] -> c[19];
rz(5.708875117102223) q[15];
rz(1.0841881442574473) q[12];
rz(4.678769092881051) q[4];
rz(0.3216831689479786) q[17];
rz(2.8161655070659863) q[19];
rz(6.088379440699022) q[13];
rz(1.7709615113035204) q[6];
cx q[18], q[8];
cx q[11], q[0];
rz(5.27663260397012) q[10];
cx q[15], q[7];
rz(3.5062786242044357) q[2];
rz(2.2466731307221184) q[1];
rz(5.332005309070663) q[9];
rz(2.6190290830555716) q[5];
rz(4.857801339067213) q[14];
rz(3.5697342700760664) q[3];
rz(3.434049480734675) q[16];
rz(3.315453178952678) q[14];
rz(2.0482285604437296) q[3];
cx q[13], q[17];
rz(4.868738522577973) q[8];
rz(3.8194094515584696) q[10];
rz(3.123198832445316) q[11];
rz(5.119027548953142) q[6];
cx q[18], q[0];
rz(5.918049932275182) q[1];
cx q[7], q[16];
cx q[5], q[4];
cx q[19], q[9];

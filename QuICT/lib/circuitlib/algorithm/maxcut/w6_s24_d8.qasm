OPENQASM 2.0;
include "qelib1.inc";
qreg q[6];
creg c[6];
h q[0];
h q[1];
h q[2];
h q[3];
h q[4];
h q[5];
cx q[5], q[2];
rz(-0.9006524085998535) q[2];
cx q[5], q[2];
cx q[3], q[1];
rz(-0.9006524085998535) q[1];
cx q[3], q[1];
cx q[3], q[2];
rz(-0.9006524085998535) q[2];
cx q[3], q[2];
cx q[0], q[5];
rz(-0.9006524085998535) q[5];
cx q[0], q[5];
rx(1.8690723180770874) q[0];
rx(1.8690723180770874) q[1];
rx(1.8690723180770874) q[2];
rx(1.8690723180770874) q[3];
rx(1.8690723180770874) q[4];
rx(1.8690723180770874) q[5];
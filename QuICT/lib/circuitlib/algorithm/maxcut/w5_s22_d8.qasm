OPENQASM 2.0;
include "qelib1.inc";
qreg q[5];
creg c[5];
h q[0];
h q[1];
h q[2];
h q[3];
h q[4];
cx q[3], q[1];
rz(-1.5747355222702026) q[1];
cx q[3], q[1];
cx q[4], q[2];
rz(-1.5747355222702026) q[2];
cx q[4], q[2];
cx q[2], q[3];
rz(-1.5747355222702026) q[3];
cx q[2], q[3];
cx q[0], q[4];
rz(-1.5747355222702026) q[4];
cx q[0], q[4];
rx(0.03889884427189827) q[0];
rx(0.03889884427189827) q[1];
rx(0.03889884427189827) q[2];
rx(0.03889884427189827) q[3];
rx(0.03889884427189827) q[4];
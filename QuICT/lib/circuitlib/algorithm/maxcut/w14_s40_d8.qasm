OPENQASM 2.0;
include "qelib1.inc";
qreg q[14];
creg c[14];
h q[0];
h q[1];
h q[2];
h q[3];
h q[4];
h q[5];
h q[6];
h q[7];
h q[8];
h q[9];
h q[10];
h q[11];
h q[12];
h q[13];
cx q[6], q[10];
rz(-0.7789721488952637) q[10];
cx q[6], q[10];
cx q[11], q[1];
rz(-0.7789721488952637) q[1];
cx q[11], q[1];
cx q[1], q[3];
rz(-0.7789721488952637) q[3];
cx q[1], q[3];
cx q[13], q[4];
rz(-0.7789721488952637) q[4];
cx q[13], q[4];
rx(0.6490803360939026) q[0];
rx(0.6490803360939026) q[1];
rx(0.6490803360939026) q[2];
rx(0.6490803360939026) q[3];
rx(0.6490803360939026) q[4];
rx(0.6490803360939026) q[5];
rx(0.6490803360939026) q[6];
rx(0.6490803360939026) q[7];
rx(0.6490803360939026) q[8];
rx(0.6490803360939026) q[9];
rx(0.6490803360939026) q[10];
rx(0.6490803360939026) q[11];
rx(0.6490803360939026) q[12];
rx(0.6490803360939026) q[13];

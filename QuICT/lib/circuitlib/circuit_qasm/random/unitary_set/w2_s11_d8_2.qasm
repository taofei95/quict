OPENQASM 2.0;
include "qelib1.inc";
qreg q[2];
creg c[2];
unitary q[0];
unitary q[1];
cx q[1], q[0];
rz(-0.5651600742614376) q[0];
ry(1.6309279853291887) q[1];
cx q[0], q[1];
ry(-1.8030298478254998) q[1];
cx q[1], q[0];
unitary q[0];
unitary q[1];
phase((0.16739075170773496+3.879033249145193e-16j)) q[0];

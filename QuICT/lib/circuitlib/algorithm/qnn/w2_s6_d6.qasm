OPENQASM 2.0;
include "qelib1.inc";
qreg q[2];
creg c[2];
x q[0];
h q[0];
rxx(0.6590526103973389) q[0], q[1];
rzz(0.9711875319480896) q[0], q[1];
rzx(0.7902168035507202) q[0], q[1];
h q[0];
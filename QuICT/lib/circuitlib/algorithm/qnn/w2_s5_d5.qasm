OPENQASM 2.0;
include "qelib1.inc";
qreg q[2];
creg c[2];
x q[0];
h q[0];
rzz(0.5047044157981873) q[0], q[1];
rzz(0.6730337142944336) q[0], q[1];
h q[0];

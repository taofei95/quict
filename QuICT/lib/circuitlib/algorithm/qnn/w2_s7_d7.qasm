OPENQASM 2.0;
include "qelib1.inc";
qreg q[2];
creg c[2];
x q[0];
x q[0];
h q[0];
rxx(0.9066019058227539) q[0], q[1];
ryy(0.25077635049819946) q[0], q[1];
rzx(0.9133961200714111) q[0], q[1];
h q[0];
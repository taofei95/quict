OPENQASM 2.0;
include "qelib1.inc";
qreg q[2];
creg c[0];
h q[0];
crz(1.5707963267948966) q[1], q[0];
h q[1];

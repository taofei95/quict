OPENQASM 2.0;
include "qelib1.inc";
qreg q[5];
creg c[5];
x q[0];
x q[2];
x q[0];
h q[0];
rxx(0.6118981838226318) q[0], q[4];
rxx(0.5535426735877991) q[1], q[4];
rxx(0.729741632938385) q[2], q[4];
rxx(0.023509681224822998) q[3], q[4];
rzx(0.4078698754310608) q[0], q[4];
rzx(0.5972901582717896) q[1], q[4];
rzx(0.9071407318115234) q[2], q[4];
rzx(0.6594429016113281) q[3], q[4];
h q[0];
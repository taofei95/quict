OPENQASM 2.0;
include "qelib1.inc";
qreg q[5];
creg c[5];
sx q[1];
sx q[2];
sy q[3];
sw q[4];
fsim(1.5707963267948966, 0) q[1], q[0];
fsim(1.5707963267948966, 0) q[3], q[2];
ry(1.5707963267948966) q[4];
sy q[1];
sx q[1];
sy q[3];
sy q[2];
sy q[0];
ry(1.5707963267948966) q[1];
sw q[1];
ry(1.5707963267948966) q[0];
rx(1.5707963267948966) q[0];
rx(1.5707963267948966) q[4];
sy q[3];
sw q[4];
sy q[3];
ry(1.5707963267948966) q[2];
ry(1.5707963267948966) q[4];
fsim(1.5707963267948966, 0) q[3], q[2];
sy q[0];
sx q[1];

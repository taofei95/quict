OPENQASM 2.0;
include "qelib1.inc";
qreg q[5];
creg c[5];
fsim(1.5707963267948966, 0) q[1], q[3];
sx q[1];
sx q[3];
sx q[2];
rx(1.5707963267948966) q[0];
fsim(1.5707963267948966, 0) q[0], q[2];
fsim(1.5707963267948966, 0) q[1], q[2];
rx(1.5707963267948966) q[4];
fsim(1.5707963267948966, 0) q[4], q[2];
sx q[4];
rx(1.5707963267948966) q[0];
fsim(1.5707963267948966, 0) q[2], q[0];
sx q[0];
fsim(1.5707963267948966, 0) q[0], q[3];
rx(1.5707963267948966) q[1];
fsim(1.5707963267948966, 0) q[1], q[3];
sy q[3];
fsim(1.5707963267948966, 0) q[3], q[4];
sx q[2];
sx q[1];
sy q[1];
sy q[1];
sy q[3];
sy q[3];
rx(1.5707963267948966) q[4];
ry(1.5707963267948966) q[1];
ry(1.5707963267948966) q[0];
fsim(1.5707963267948966, 0) q[4], q[3];
sw q[3];
sx q[2];
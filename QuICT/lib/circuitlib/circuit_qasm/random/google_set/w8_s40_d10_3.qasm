OPENQASM 2.0;
include "qelib1.inc";
qreg q[8];
creg c[8];
rx(1.5707963267948966) q[6];
fsim(1.5707963267948966, 0) q[1], q[6];
sx q[5];
sx q[5];
ry(1.5707963267948966) q[1];
rx(1.5707963267948966) q[7];
ry(1.5707963267948966) q[3];
fsim(1.5707963267948966, 0) q[4], q[0];
sx q[0];
fsim(1.5707963267948966, 0) q[4], q[0];
rx(1.5707963267948966) q[0];
sx q[4];
ry(1.5707963267948966) q[5];
fsim(1.5707963267948966, 0) q[5], q[4];
fsim(1.5707963267948966, 0) q[6], q[3];
rx(1.5707963267948966) q[4];
sw q[4];
fsim(1.5707963267948966, 0) q[7], q[6];
rx(1.5707963267948966) q[3];
fsim(1.5707963267948966, 0) q[3], q[0];
ry(1.5707963267948966) q[1];
sw q[5];
fsim(1.5707963267948966, 0) q[0], q[1];
sx q[5];
fsim(1.5707963267948966, 0) q[7], q[0];
sw q[3];
sx q[4];
ry(1.5707963267948966) q[2];
fsim(1.5707963267948966, 0) q[3], q[5];
ry(1.5707963267948966) q[2];
rx(1.5707963267948966) q[2];
sx q[3];
ry(1.5707963267948966) q[4];
sw q[6];
sw q[5];
ry(1.5707963267948966) q[6];
rx(1.5707963267948966) q[0];
sy q[1];
sw q[4];
fsim(1.5707963267948966, 0) q[0], q[6];

OPENQASM 2.0;
include "qelib1.inc";
qreg q[13];
creg c[13];
ry(1.5707963267948966) q[10];
fsim(1.5707963267948966, 0) q[3], q[11];
rx(1.5707963267948966) q[6];
ry(1.5707963267948966) q[4];
rx(1.5707963267948966) q[0];
sw q[6];
fsim(1.5707963267948966, 0) q[10], q[12];
rx(1.5707963267948966) q[11];
rx(1.5707963267948966) q[7];
ry(1.5707963267948966) q[2];
ry(1.5707963267948966) q[11];
sw q[11];
fsim(1.5707963267948966, 0) q[6], q[11];
sy q[1];
ry(1.5707963267948966) q[6];
sx q[8];
sw q[11];
fsim(1.5707963267948966, 0) q[7], q[4];
fsim(1.5707963267948966, 0) q[5], q[9];
rx(1.5707963267948966) q[2];
ry(1.5707963267948966) q[5];
sx q[6];
sw q[0];
sx q[5];
sx q[1];
sx q[12];
fsim(1.5707963267948966, 0) q[6], q[11];
fsim(1.5707963267948966, 0) q[6], q[11];
rx(1.5707963267948966) q[7];
sw q[1];
fsim(1.5707963267948966, 0) q[1], q[2];
fsim(1.5707963267948966, 0) q[11], q[7];
sx q[4];
ry(1.5707963267948966) q[11];
rx(1.5707963267948966) q[8];
sx q[3];
sw q[6];
ry(1.5707963267948966) q[4];
sx q[7];
sx q[2];
fsim(1.5707963267948966, 0) q[3], q[8];
fsim(1.5707963267948966, 0) q[1], q[6];
sx q[5];
ry(1.5707963267948966) q[9];
ry(1.5707963267948966) q[0];
sx q[4];
sw q[0];
sx q[9];
rx(1.5707963267948966) q[5];
fsim(1.5707963267948966, 0) q[12], q[10];
ry(1.5707963267948966) q[6];
sw q[6];
rx(1.5707963267948966) q[5];
fsim(1.5707963267948966, 0) q[7], q[5];
sx q[4];
rx(1.5707963267948966) q[2];
sx q[8];
sw q[0];
fsim(1.5707963267948966, 0) q[7], q[3];
ry(1.5707963267948966) q[2];
rx(1.5707963267948966) q[5];
fsim(1.5707963267948966, 0) q[2], q[1];
fsim(1.5707963267948966, 0) q[7], q[3];
fsim(1.5707963267948966, 0) q[3], q[7];
ry(1.5707963267948966) q[9];
rx(1.5707963267948966) q[8];
sw q[5];
ry(1.5707963267948966) q[8];
rx(1.5707963267948966) q[9];
sw q[6];
fsim(1.5707963267948966, 0) q[7], q[2];
sw q[12];
rx(1.5707963267948966) q[6];
sw q[6];
sw q[2];
sw q[8];
sx q[9];
sx q[11];

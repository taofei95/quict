OPENQASM 2.0;
include "qelib1.inc";
qreg q[12];
creg c[12];
ry(1.5707963267948966) q[11];
ry(1.5707963267948966) q[7];
rx(1.5707963267948966) q[9];
fsim(1.5707963267948966, 0) q[10], q[2];
sx q[1];
fsim(1.5707963267948966, 0) q[7], q[3];
ry(1.5707963267948966) q[5];
ry(1.5707963267948966) q[10];
sw q[6];
fsim(1.5707963267948966, 0) q[11], q[5];
sw q[9];
sx q[0];
sx q[4];
sy q[8];
ry(1.5707963267948966) q[11];
sy q[11];
ry(1.5707963267948966) q[3];
fsim(1.5707963267948966, 0) q[1], q[8];
fsim(1.5707963267948966, 0) q[8], q[0];
fsim(1.5707963267948966, 0) q[7], q[8];
ry(1.5707963267948966) q[6];
fsim(1.5707963267948966, 0) q[11], q[2];
sw q[9];
sy q[9];
fsim(1.5707963267948966, 0) q[6], q[9];
sx q[7];
rx(1.5707963267948966) q[1];
sy q[10];
sw q[4];
ry(1.5707963267948966) q[3];
fsim(1.5707963267948966, 0) q[11], q[4];
sy q[8];
sx q[4];
sy q[5];
sw q[8];
fsim(1.5707963267948966, 0) q[9], q[0];
fsim(1.5707963267948966, 0) q[4], q[1];
sw q[5];
fsim(1.5707963267948966, 0) q[3], q[11];
sy q[8];
sy q[9];
ry(1.5707963267948966) q[3];
sw q[5];
sx q[6];
sy q[10];
ry(1.5707963267948966) q[9];
rx(1.5707963267948966) q[0];
ry(1.5707963267948966) q[3];
ry(1.5707963267948966) q[0];
sw q[1];
fsim(1.5707963267948966, 0) q[1], q[5];
sx q[6];
ry(1.5707963267948966) q[9];
sy q[7];
sx q[11];
fsim(1.5707963267948966, 0) q[9], q[1];
sw q[1];
ry(1.5707963267948966) q[1];
ry(1.5707963267948966) q[6];
sw q[9];
sx q[5];
sx q[4];
sw q[8];
sw q[5];
sx q[0];
ry(1.5707963267948966) q[1];
rx(1.5707963267948966) q[11];
rx(1.5707963267948966) q[11];
rx(1.5707963267948966) q[8];
ry(1.5707963267948966) q[3];
sw q[0];
rx(1.5707963267948966) q[0];
sy q[8];
rx(1.5707963267948966) q[9];
sx q[5];
fsim(1.5707963267948966, 0) q[10], q[9];
fsim(1.5707963267948966, 0) q[8], q[1];
rx(1.5707963267948966) q[2];
sy q[4];
fsim(1.5707963267948966, 0) q[2], q[4];
sy q[5];
fsim(1.5707963267948966, 0) q[4], q[3];
sy q[3];
sy q[7];
sw q[7];
sw q[4];
sy q[9];
ry(1.5707963267948966) q[2];
rx(1.5707963267948966) q[3];
sx q[8];
rx(1.5707963267948966) q[5];
rx(1.5707963267948966) q[7];
rx(1.5707963267948966) q[11];
fsim(1.5707963267948966, 0) q[4], q[5];
sy q[7];
sy q[10];
sx q[11];
sy q[5];
ry(1.5707963267948966) q[3];
ry(1.5707963267948966) q[6];
ry(1.5707963267948966) q[9];
sw q[3];
rx(1.5707963267948966) q[2];
sy q[0];
sy q[10];
sw q[5];
fsim(1.5707963267948966, 0) q[3], q[2];
sw q[7];
sx q[0];
sw q[2];
sy q[11];
sw q[5];
sx q[2];
fsim(1.5707963267948966, 0) q[0], q[2];
fsim(1.5707963267948966, 0) q[11], q[10];
rx(1.5707963267948966) q[10];
rx(1.5707963267948966) q[6];
fsim(1.5707963267948966, 0) q[0], q[1];
ry(1.5707963267948966) q[0];
sy q[9];

OPENQASM 2.0;
include "qelib1.inc";
qreg q[13];
creg c[13];
sy q[2];
sx q[5];
sy q[7];
ry(1.5707963267948966) q[0];
ry(1.5707963267948966) q[12];
sy q[7];
fsim(1.5707963267948966, 0) q[8], q[10];
ry(1.5707963267948966) q[5];
sx q[5];
sw q[2];
sw q[11];
sx q[1];
fsim(1.5707963267948966, 0) q[3], q[8];
ry(1.5707963267948966) q[10];
sx q[9];
fsim(1.5707963267948966, 0) q[7], q[2];
rx(1.5707963267948966) q[5];
sw q[5];
sw q[1];
sw q[3];
sy q[7];
rx(1.5707963267948966) q[11];
rx(1.5707963267948966) q[4];
fsim(1.5707963267948966, 0) q[7], q[11];
sx q[0];
ry(1.5707963267948966) q[10];
sy q[11];
sw q[4];
sw q[4];
rx(1.5707963267948966) q[0];
fsim(1.5707963267948966, 0) q[1], q[9];
ry(1.5707963267948966) q[3];
ry(1.5707963267948966) q[2];
rx(1.5707963267948966) q[11];
ry(1.5707963267948966) q[7];
fsim(1.5707963267948966, 0) q[6], q[4];
sx q[11];
rx(1.5707963267948966) q[7];
ry(1.5707963267948966) q[0];
ry(1.5707963267948966) q[9];
rx(1.5707963267948966) q[8];
sx q[0];
fsim(1.5707963267948966, 0) q[7], q[6];
sw q[4];
sw q[1];
sy q[1];
fsim(1.5707963267948966, 0) q[8], q[3];
fsim(1.5707963267948966, 0) q[7], q[8];
sx q[8];
rx(1.5707963267948966) q[7];
sx q[0];
fsim(1.5707963267948966, 0) q[12], q[1];
rx(1.5707963267948966) q[10];
sx q[8];
sx q[3];
rx(1.5707963267948966) q[10];
ry(1.5707963267948966) q[11];
rx(1.5707963267948966) q[3];
sw q[11];
sx q[6];
sy q[4];
sw q[6];
sw q[4];
rx(1.5707963267948966) q[7];
fsim(1.5707963267948966, 0) q[11], q[5];

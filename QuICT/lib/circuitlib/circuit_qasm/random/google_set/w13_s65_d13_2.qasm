OPENQASM 2.0;
include "qelib1.inc";
qreg q[13];
creg c[13];
ry(1.5707963267948966) q[6];
rx(1.5707963267948966) q[2];
ry(1.5707963267948966) q[4];
sx q[12];
ry(1.5707963267948966) q[12];
sx q[2];
rx(1.5707963267948966) q[9];
fsim(1.5707963267948966, 0) q[2], q[12];
sy q[5];
rx(1.5707963267948966) q[9];
rx(1.5707963267948966) q[6];
sw q[7];
sy q[7];
sx q[2];
ry(1.5707963267948966) q[9];
rx(1.5707963267948966) q[4];
ry(1.5707963267948966) q[12];
fsim(1.5707963267948966, 0) q[12], q[11];
sy q[0];
sx q[1];
sw q[6];
rx(1.5707963267948966) q[3];
sy q[12];
fsim(1.5707963267948966, 0) q[0], q[7];
rx(1.5707963267948966) q[5];
sy q[5];
sx q[7];
sy q[0];
sy q[10];
fsim(1.5707963267948966, 0) q[4], q[6];
fsim(1.5707963267948966, 0) q[9], q[12];
ry(1.5707963267948966) q[2];
sy q[2];
ry(1.5707963267948966) q[4];
fsim(1.5707963267948966, 0) q[4], q[3];
fsim(1.5707963267948966, 0) q[12], q[5];
sy q[11];
sw q[3];
fsim(1.5707963267948966, 0) q[4], q[3];
fsim(1.5707963267948966, 0) q[6], q[3];
ry(1.5707963267948966) q[11];
fsim(1.5707963267948966, 0) q[2], q[11];
ry(1.5707963267948966) q[8];
sy q[5];
sw q[2];
fsim(1.5707963267948966, 0) q[2], q[0];
sy q[12];
sy q[4];
sy q[7];
fsim(1.5707963267948966, 0) q[3], q[4];
fsim(1.5707963267948966, 0) q[1], q[9];
sx q[11];
fsim(1.5707963267948966, 0) q[9], q[3];
ry(1.5707963267948966) q[7];
rx(1.5707963267948966) q[6];
fsim(1.5707963267948966, 0) q[3], q[5];
sx q[0];
sy q[1];
rx(1.5707963267948966) q[7];
ry(1.5707963267948966) q[6];
rx(1.5707963267948966) q[1];
fsim(1.5707963267948966, 0) q[0], q[7];
fsim(1.5707963267948966, 0) q[4], q[2];
sy q[12];
sy q[0];

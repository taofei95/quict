OPENQASM 2.0;
include "qelib1.inc";
qreg q[17];
creg c[17];
ry(1.5707963267948966) q[1];
sw q[1];
ry(1.5707963267948966) q[0];
rx(1.5707963267948966) q[5];
rx(1.5707963267948966) q[10];
fsim(1.5707963267948966, 0) q[4], q[10];
ry(1.5707963267948966) q[14];
rx(1.5707963267948966) q[9];
ry(1.5707963267948966) q[10];
sw q[4];
rx(1.5707963267948966) q[16];
sw q[12];
sy q[8];
sx q[13];
sx q[6];
rx(1.5707963267948966) q[5];
sw q[6];
sy q[9];
rx(1.5707963267948966) q[10];
sx q[12];
fsim(1.5707963267948966, 0) q[16], q[4];
sy q[1];
ry(1.5707963267948966) q[8];
sw q[14];
fsim(1.5707963267948966, 0) q[8], q[0];
sx q[4];
fsim(1.5707963267948966, 0) q[1], q[4];
fsim(1.5707963267948966, 0) q[10], q[1];
sw q[10];
ry(1.5707963267948966) q[1];
fsim(1.5707963267948966, 0) q[6], q[12];
ry(1.5707963267948966) q[16];
rx(1.5707963267948966) q[0];
sw q[11];
ry(1.5707963267948966) q[6];
rx(1.5707963267948966) q[12];
sy q[4];
ry(1.5707963267948966) q[11];
sw q[2];
sy q[5];
ry(1.5707963267948966) q[3];
sy q[16];
fsim(1.5707963267948966, 0) q[6], q[13];
fsim(1.5707963267948966, 0) q[1], q[8];
ry(1.5707963267948966) q[16];
ry(1.5707963267948966) q[0];
rx(1.5707963267948966) q[0];
ry(1.5707963267948966) q[0];
rx(1.5707963267948966) q[13];
ry(1.5707963267948966) q[1];
ry(1.5707963267948966) q[9];
fsim(1.5707963267948966, 0) q[0], q[9];
sy q[16];
fsim(1.5707963267948966, 0) q[8], q[5];
ry(1.5707963267948966) q[3];
sy q[6];
rx(1.5707963267948966) q[10];
ry(1.5707963267948966) q[16];
ry(1.5707963267948966) q[12];
sw q[15];
ry(1.5707963267948966) q[0];
fsim(1.5707963267948966, 0) q[1], q[7];
sw q[6];
sy q[11];
ry(1.5707963267948966) q[5];
ry(1.5707963267948966) q[3];
rx(1.5707963267948966) q[11];
ry(1.5707963267948966) q[11];
sx q[1];
rx(1.5707963267948966) q[13];
sw q[10];
fsim(1.5707963267948966, 0) q[5], q[11];
fsim(1.5707963267948966, 0) q[14], q[8];
ry(1.5707963267948966) q[9];
sy q[3];
fsim(1.5707963267948966, 0) q[2], q[15];
sx q[5];
sy q[10];
fsim(1.5707963267948966, 0) q[11], q[4];
sw q[3];
sx q[9];
fsim(1.5707963267948966, 0) q[9], q[13];
sw q[2];
ry(1.5707963267948966) q[11];
sw q[14];

OPENQASM 2.0;
include "qelib1.inc";
qreg q[18];
creg c[18];
sy q[9];
ry(1.5707963267948966) q[10];
sx q[10];
rx(1.5707963267948966) q[0];
ry(1.5707963267948966) q[0];
rx(1.5707963267948966) q[0];
fsim(1.5707963267948966, 0) q[12], q[5];
ry(1.5707963267948966) q[3];
sw q[17];
ry(1.5707963267948966) q[8];
sx q[4];
rx(1.5707963267948966) q[0];
rx(1.5707963267948966) q[5];
sy q[5];
fsim(1.5707963267948966, 0) q[7], q[9];
sy q[5];
fsim(1.5707963267948966, 0) q[16], q[10];
fsim(1.5707963267948966, 0) q[15], q[11];
fsim(1.5707963267948966, 0) q[15], q[14];
fsim(1.5707963267948966, 0) q[3], q[8];
sw q[9];
rx(1.5707963267948966) q[8];
rx(1.5707963267948966) q[5];
fsim(1.5707963267948966, 0) q[1], q[5];
sx q[14];
sx q[10];
rx(1.5707963267948966) q[11];
ry(1.5707963267948966) q[14];
sy q[9];
sy q[13];
ry(1.5707963267948966) q[14];
rx(1.5707963267948966) q[14];
ry(1.5707963267948966) q[1];
fsim(1.5707963267948966, 0) q[14], q[9];
rx(1.5707963267948966) q[7];
sy q[11];
ry(1.5707963267948966) q[8];
rx(1.5707963267948966) q[4];
sx q[3];
sy q[4];
fsim(1.5707963267948966, 0) q[2], q[0];
ry(1.5707963267948966) q[13];
ry(1.5707963267948966) q[1];
rx(1.5707963267948966) q[5];
ry(1.5707963267948966) q[5];
rx(1.5707963267948966) q[14];
rx(1.5707963267948966) q[10];
sw q[15];
fsim(1.5707963267948966, 0) q[16], q[12];
rx(1.5707963267948966) q[11];
ry(1.5707963267948966) q[0];
sx q[13];
rx(1.5707963267948966) q[12];
fsim(1.5707963267948966, 0) q[4], q[13];
sx q[8];
ry(1.5707963267948966) q[3];
sx q[0];
fsim(1.5707963267948966, 0) q[13], q[17];
sw q[2];
rx(1.5707963267948966) q[17];
fsim(1.5707963267948966, 0) q[1], q[17];
fsim(1.5707963267948966, 0) q[12], q[3];
ry(1.5707963267948966) q[12];
sw q[10];
sw q[3];
rx(1.5707963267948966) q[6];
sx q[15];
fsim(1.5707963267948966, 0) q[10], q[0];
sx q[1];
sx q[2];
fsim(1.5707963267948966, 0) q[7], q[17];
rx(1.5707963267948966) q[11];
ry(1.5707963267948966) q[10];
sy q[0];
fsim(1.5707963267948966, 0) q[11], q[0];
sy q[6];
ry(1.5707963267948966) q[17];
rx(1.5707963267948966) q[0];
sx q[2];
sw q[9];
rx(1.5707963267948966) q[9];
fsim(1.5707963267948966, 0) q[7], q[3];
sy q[13];
sw q[13];
fsim(1.5707963267948966, 0) q[8], q[2];
fsim(1.5707963267948966, 0) q[10], q[5];
rx(1.5707963267948966) q[4];
sw q[16];
sx q[5];
sw q[13];
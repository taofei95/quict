OPENQASM 2.0;
include "qelib1.inc";
qreg q[17];
creg c[17];
sx q[1];
rx(1.5707963267948966) q[15];
sy q[6];
fsim(1.5707963267948966, 0) q[8], q[3];
sw q[1];
rx(1.5707963267948966) q[12];
sy q[14];
sw q[2];
sy q[6];
sx q[4];
sx q[12];
sy q[9];
sx q[2];
sx q[10];
fsim(1.5707963267948966, 0) q[2], q[5];
rx(1.5707963267948966) q[2];
sx q[7];
sw q[3];
rx(1.5707963267948966) q[14];
fsim(1.5707963267948966, 0) q[10], q[2];
sx q[5];
sy q[14];
fsim(1.5707963267948966, 0) q[8], q[7];
fsim(1.5707963267948966, 0) q[2], q[11];
sw q[6];
sw q[1];
sy q[10];
sy q[13];
sw q[1];
ry(1.5707963267948966) q[3];
ry(1.5707963267948966) q[10];
fsim(1.5707963267948966, 0) q[11], q[13];
sx q[4];
ry(1.5707963267948966) q[4];
fsim(1.5707963267948966, 0) q[5], q[1];
rx(1.5707963267948966) q[1];
rx(1.5707963267948966) q[5];
fsim(1.5707963267948966, 0) q[6], q[3];
ry(1.5707963267948966) q[13];
sy q[4];
sy q[2];
rx(1.5707963267948966) q[12];
fsim(1.5707963267948966, 0) q[2], q[0];
rx(1.5707963267948966) q[9];
sw q[2];
sw q[4];
fsim(1.5707963267948966, 0) q[0], q[1];
rx(1.5707963267948966) q[9];
ry(1.5707963267948966) q[11];
fsim(1.5707963267948966, 0) q[11], q[4];
sw q[10];
fsim(1.5707963267948966, 0) q[15], q[16];
fsim(1.5707963267948966, 0) q[4], q[0];
sx q[5];
ry(1.5707963267948966) q[7];
sx q[9];
sx q[0];
sx q[0];
fsim(1.5707963267948966, 0) q[15], q[16];
sy q[8];
sx q[5];
rx(1.5707963267948966) q[3];
ry(1.5707963267948966) q[7];
sx q[13];
sy q[11];
rx(1.5707963267948966) q[5];
ry(1.5707963267948966) q[6];
sw q[5];
sy q[9];
ry(1.5707963267948966) q[16];
fsim(1.5707963267948966, 0) q[7], q[5];
sw q[11];
sx q[5];
sy q[7];
sw q[8];
sw q[5];
sx q[8];
sx q[8];
rx(1.5707963267948966) q[0];
rx(1.5707963267948966) q[12];
fsim(1.5707963267948966, 0) q[14], q[3];
sx q[11];
fsim(1.5707963267948966, 0) q[6], q[11];
ry(1.5707963267948966) q[3];
sx q[12];

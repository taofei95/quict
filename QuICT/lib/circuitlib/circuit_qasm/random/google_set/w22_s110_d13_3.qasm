OPENQASM 2.0;
include "qelib1.inc";
qreg q[22];
creg c[22];
fsim(1.5707963267948966, 0) q[15], q[0];
sw q[4];
ry(1.5707963267948966) q[18];
sy q[16];
rx(1.5707963267948966) q[21];
ry(1.5707963267948966) q[2];
fsim(1.5707963267948966, 0) q[11], q[2];
sx q[8];
rx(1.5707963267948966) q[14];
sy q[13];
sw q[11];
fsim(1.5707963267948966, 0) q[3], q[4];
sy q[4];
ry(1.5707963267948966) q[16];
sw q[17];
sw q[12];
fsim(1.5707963267948966, 0) q[20], q[5];
fsim(1.5707963267948966, 0) q[14], q[1];
rx(1.5707963267948966) q[14];
sy q[6];
sx q[2];
rx(1.5707963267948966) q[21];
sx q[13];
sw q[0];
sw q[6];
fsim(1.5707963267948966, 0) q[16], q[19];
sw q[12];
fsim(1.5707963267948966, 0) q[14], q[8];
sw q[1];
rx(1.5707963267948966) q[0];
fsim(1.5707963267948966, 0) q[5], q[9];
sw q[4];
ry(1.5707963267948966) q[4];
sx q[5];
ry(1.5707963267948966) q[15];
ry(1.5707963267948966) q[4];
sy q[4];
ry(1.5707963267948966) q[4];
sy q[0];
sw q[13];
fsim(1.5707963267948966, 0) q[11], q[2];
sx q[9];
sw q[15];
sw q[19];
fsim(1.5707963267948966, 0) q[3], q[16];
fsim(1.5707963267948966, 0) q[0], q[11];
rx(1.5707963267948966) q[21];
sx q[13];
sw q[0];
fsim(1.5707963267948966, 0) q[0], q[10];
ry(1.5707963267948966) q[2];
fsim(1.5707963267948966, 0) q[9], q[20];
fsim(1.5707963267948966, 0) q[20], q[12];
sy q[9];
sw q[8];
sy q[17];
sx q[18];
fsim(1.5707963267948966, 0) q[15], q[19];
sx q[11];
sw q[19];
ry(1.5707963267948966) q[16];
sx q[17];
rx(1.5707963267948966) q[21];
fsim(1.5707963267948966, 0) q[7], q[17];
sy q[8];
sx q[21];
ry(1.5707963267948966) q[12];
rx(1.5707963267948966) q[16];
ry(1.5707963267948966) q[5];
sw q[1];
fsim(1.5707963267948966, 0) q[10], q[6];
rx(1.5707963267948966) q[2];
fsim(1.5707963267948966, 0) q[12], q[16];
fsim(1.5707963267948966, 0) q[3], q[18];
fsim(1.5707963267948966, 0) q[20], q[9];
sw q[0];
fsim(1.5707963267948966, 0) q[16], q[7];
sx q[11];
sy q[10];
fsim(1.5707963267948966, 0) q[6], q[1];
rx(1.5707963267948966) q[6];
sy q[16];
ry(1.5707963267948966) q[12];
rx(1.5707963267948966) q[7];
sx q[9];
sw q[12];
ry(1.5707963267948966) q[16];
fsim(1.5707963267948966, 0) q[11], q[15];
sy q[8];
sy q[12];
ry(1.5707963267948966) q[15];
fsim(1.5707963267948966, 0) q[2], q[20];
rx(1.5707963267948966) q[14];
rx(1.5707963267948966) q[3];
fsim(1.5707963267948966, 0) q[9], q[2];
sw q[21];
sy q[15];
rx(1.5707963267948966) q[19];
sx q[21];
sw q[17];
sy q[15];
rx(1.5707963267948966) q[20];
rx(1.5707963267948966) q[0];
fsim(1.5707963267948966, 0) q[10], q[0];
ry(1.5707963267948966) q[10];
sw q[10];
sw q[21];
rx(1.5707963267948966) q[1];
sx q[10];
fsim(1.5707963267948966, 0) q[2], q[6];

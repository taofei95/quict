OPENQASM 2.0;
include "qelib1.inc";
qreg q[26];
creg c[26];
ry(1.5707963267948966) q[17];
fsim(1.5707963267948966, 0) q[20], q[6];
sw q[12];
fsim(1.5707963267948966, 0) q[12], q[14];
fsim(1.5707963267948966, 0) q[21], q[13];
rx(1.5707963267948966) q[22];
fsim(1.5707963267948966, 0) q[18], q[25];
sy q[3];
sx q[25];
fsim(1.5707963267948966, 0) q[11], q[9];
fsim(1.5707963267948966, 0) q[23], q[0];
ry(1.5707963267948966) q[19];
rx(1.5707963267948966) q[6];
sw q[14];
sx q[18];
rx(1.5707963267948966) q[21];
ry(1.5707963267948966) q[9];
ry(1.5707963267948966) q[7];
rx(1.5707963267948966) q[2];
sy q[13];
sy q[5];
sw q[13];
sy q[11];
fsim(1.5707963267948966, 0) q[1], q[6];
sx q[16];
sw q[2];
rx(1.5707963267948966) q[22];
fsim(1.5707963267948966, 0) q[5], q[15];
ry(1.5707963267948966) q[1];
ry(1.5707963267948966) q[6];
sy q[2];
ry(1.5707963267948966) q[11];
sw q[9];
sw q[18];
ry(1.5707963267948966) q[13];
sy q[16];
fsim(1.5707963267948966, 0) q[24], q[12];
sx q[6];
sy q[24];
sy q[14];
sx q[22];
rx(1.5707963267948966) q[22];
fsim(1.5707963267948966, 0) q[2], q[14];
sw q[24];
sw q[8];
rx(1.5707963267948966) q[5];
fsim(1.5707963267948966, 0) q[1], q[16];
sy q[5];
sy q[13];
sy q[22];
sy q[9];
rx(1.5707963267948966) q[15];
sy q[20];
sx q[3];
sw q[1];
fsim(1.5707963267948966, 0) q[17], q[2];
sx q[23];
sy q[0];
sy q[9];
sw q[4];
fsim(1.5707963267948966, 0) q[16], q[21];
fsim(1.5707963267948966, 0) q[23], q[16];
rx(1.5707963267948966) q[23];
fsim(1.5707963267948966, 0) q[6], q[18];
fsim(1.5707963267948966, 0) q[10], q[8];
sw q[3];
sy q[12];
sw q[21];
fsim(1.5707963267948966, 0) q[4], q[12];
sy q[2];
sw q[1];
fsim(1.5707963267948966, 0) q[11], q[17];
sy q[21];
rx(1.5707963267948966) q[24];
sy q[18];
sx q[1];
sw q[7];
sy q[18];
sw q[23];
sy q[13];
sy q[5];
rx(1.5707963267948966) q[25];
fsim(1.5707963267948966, 0) q[2], q[15];
rx(1.5707963267948966) q[4];
sw q[21];
ry(1.5707963267948966) q[9];
rx(1.5707963267948966) q[17];
sx q[6];
fsim(1.5707963267948966, 0) q[7], q[1];
ry(1.5707963267948966) q[18];
sy q[23];
ry(1.5707963267948966) q[10];
ry(1.5707963267948966) q[17];
rx(1.5707963267948966) q[14];
rx(1.5707963267948966) q[16];
fsim(1.5707963267948966, 0) q[0], q[23];
sw q[15];
rx(1.5707963267948966) q[12];
ry(1.5707963267948966) q[18];
sy q[12];
rx(1.5707963267948966) q[17];
ry(1.5707963267948966) q[16];
sx q[6];
sw q[4];
rx(1.5707963267948966) q[3];
fsim(1.5707963267948966, 0) q[16], q[13];
fsim(1.5707963267948966, 0) q[9], q[1];
fsim(1.5707963267948966, 0) q[21], q[1];
sw q[1];
fsim(1.5707963267948966, 0) q[24], q[1];
ry(1.5707963267948966) q[6];
fsim(1.5707963267948966, 0) q[2], q[11];
rx(1.5707963267948966) q[15];
ry(1.5707963267948966) q[23];
sw q[18];
ry(1.5707963267948966) q[15];
fsim(1.5707963267948966, 0) q[1], q[11];
ry(1.5707963267948966) q[1];
ry(1.5707963267948966) q[17];
ry(1.5707963267948966) q[7];
ry(1.5707963267948966) q[2];
fsim(1.5707963267948966, 0) q[0], q[15];
ry(1.5707963267948966) q[13];
sy q[5];
sy q[0];
sw q[25];
sy q[0];
rx(1.5707963267948966) q[21];
fsim(1.5707963267948966, 0) q[5], q[4];
fsim(1.5707963267948966, 0) q[8], q[16];

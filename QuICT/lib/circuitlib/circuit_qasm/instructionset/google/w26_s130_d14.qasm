OPENQASM 2.0;
include "qelib1.inc";
qreg q[26];
creg c[26];
sy q[14];
sx q[24];
sy q[19];
sx q[5];
sy q[17];
fsim(1.5707963267948966, 0) q[4], q[8];
sx q[7];
sw q[12];
rx(1.5707963267948966) q[21];
ry(1.5707963267948966) q[3];
ry(1.5707963267948966) q[23];
rx(1.5707963267948966) q[7];
sy q[24];
fsim(1.5707963267948966, 0) q[25], q[3];
sy q[15];
fsim(1.5707963267948966, 0) q[3], q[9];
sx q[10];
sw q[15];
sx q[15];
sy q[10];
sw q[16];
sy q[11];
rx(1.5707963267948966) q[5];
fsim(1.5707963267948966, 0) q[17], q[20];
sx q[19];
sx q[1];
sx q[21];
fsim(1.5707963267948966, 0) q[19], q[12];
sw q[7];
sy q[19];
ry(1.5707963267948966) q[15];
sw q[1];
sx q[21];
sy q[5];
sy q[17];
sw q[23];
fsim(1.5707963267948966, 0) q[24], q[22];
rx(1.5707963267948966) q[18];
sy q[20];
sy q[13];
fsim(1.5707963267948966, 0) q[24], q[10];
sw q[9];
rx(1.5707963267948966) q[4];
sy q[8];
sy q[8];
sy q[11];
rx(1.5707963267948966) q[15];
ry(1.5707963267948966) q[6];
sy q[23];
sy q[4];
sy q[25];
rx(1.5707963267948966) q[0];
rx(1.5707963267948966) q[23];
rx(1.5707963267948966) q[12];
fsim(1.5707963267948966, 0) q[18], q[21];
sx q[21];
sx q[22];
sy q[11];
sx q[3];
sw q[23];
fsim(1.5707963267948966, 0) q[14], q[23];
rx(1.5707963267948966) q[4];
sx q[19];
sx q[18];
ry(1.5707963267948966) q[0];
sx q[12];
sx q[15];
sw q[22];
ry(1.5707963267948966) q[13];
ry(1.5707963267948966) q[24];
sx q[5];
sw q[12];
sy q[24];
fsim(1.5707963267948966, 0) q[6], q[20];
ry(1.5707963267948966) q[17];
fsim(1.5707963267948966, 0) q[15], q[16];
rx(1.5707963267948966) q[12];
sy q[21];
rx(1.5707963267948966) q[16];
sw q[22];
fsim(1.5707963267948966, 0) q[3], q[14];
sw q[15];
sy q[18];
rx(1.5707963267948966) q[6];
rx(1.5707963267948966) q[20];
sw q[24];
fsim(1.5707963267948966, 0) q[11], q[14];
fsim(1.5707963267948966, 0) q[21], q[2];
rx(1.5707963267948966) q[21];
sy q[22];
sx q[15];
fsim(1.5707963267948966, 0) q[21], q[14];
rx(1.5707963267948966) q[24];
sw q[18];
rx(1.5707963267948966) q[0];
fsim(1.5707963267948966, 0) q[25], q[21];
fsim(1.5707963267948966, 0) q[4], q[21];
ry(1.5707963267948966) q[17];
sy q[20];
sx q[20];
sw q[11];
ry(1.5707963267948966) q[17];
rx(1.5707963267948966) q[4];
sw q[5];
ry(1.5707963267948966) q[24];
sx q[14];
sy q[25];
ry(1.5707963267948966) q[16];
rx(1.5707963267948966) q[5];
sy q[7];
ry(1.5707963267948966) q[7];
rx(1.5707963267948966) q[11];
rx(1.5707963267948966) q[14];
rx(1.5707963267948966) q[0];
fsim(1.5707963267948966, 0) q[1], q[5];
ry(1.5707963267948966) q[15];
fsim(1.5707963267948966, 0) q[15], q[4];
rx(1.5707963267948966) q[2];
rx(1.5707963267948966) q[14];
ry(1.5707963267948966) q[3];
ry(1.5707963267948966) q[3];
rx(1.5707963267948966) q[1];
rx(1.5707963267948966) q[22];
ry(1.5707963267948966) q[21];
sx q[18];
sw q[15];
sx q[8];
fsim(1.5707963267948966, 0) q[14], q[23];
ry(1.5707963267948966) q[24];
sy q[6];
OPENQASM 2.0;
include "qelib1.inc";
qreg q[28];
creg c[28];
rx(1.5707963267948966) q[19];
rx(1.5707963267948966) q[17];
rx(1.5707963267948966) q[11];
ry(1.5707963267948966) q[24];
sw q[4];
rx(1.5707963267948966) q[9];
sy q[3];
sy q[9];
sy q[3];
fsim(1.5707963267948966, 0) q[1], q[6];
ry(1.5707963267948966) q[6];
ry(1.5707963267948966) q[19];
sy q[7];
rx(1.5707963267948966) q[2];
ry(1.5707963267948966) q[0];
ry(1.5707963267948966) q[13];
fsim(1.5707963267948966, 0) q[2], q[25];
ry(1.5707963267948966) q[8];
ry(1.5707963267948966) q[5];
fsim(1.5707963267948966, 0) q[24], q[22];
sw q[13];
fsim(1.5707963267948966, 0) q[17], q[15];
sx q[5];
ry(1.5707963267948966) q[10];
sy q[4];
ry(1.5707963267948966) q[3];
rx(1.5707963267948966) q[16];
sw q[14];
sy q[14];
ry(1.5707963267948966) q[24];
ry(1.5707963267948966) q[8];
sy q[21];
ry(1.5707963267948966) q[9];
sy q[25];
fsim(1.5707963267948966, 0) q[27], q[14];
rx(1.5707963267948966) q[25];
ry(1.5707963267948966) q[25];
sw q[10];
rx(1.5707963267948966) q[21];
rx(1.5707963267948966) q[9];
sy q[20];
fsim(1.5707963267948966, 0) q[6], q[17];
sy q[14];
sw q[14];
ry(1.5707963267948966) q[12];
fsim(1.5707963267948966, 0) q[14], q[25];
sw q[2];
fsim(1.5707963267948966, 0) q[4], q[22];
fsim(1.5707963267948966, 0) q[15], q[9];
fsim(1.5707963267948966, 0) q[25], q[22];
sx q[21];
sy q[19];
fsim(1.5707963267948966, 0) q[24], q[25];
sw q[2];
fsim(1.5707963267948966, 0) q[8], q[0];
sx q[12];
ry(1.5707963267948966) q[9];
ry(1.5707963267948966) q[10];
rx(1.5707963267948966) q[15];
sy q[7];
fsim(1.5707963267948966, 0) q[21], q[5];
ry(1.5707963267948966) q[26];
fsim(1.5707963267948966, 0) q[13], q[14];
rx(1.5707963267948966) q[10];
fsim(1.5707963267948966, 0) q[13], q[18];
sw q[0];
sx q[25];
sw q[23];
sx q[1];
ry(1.5707963267948966) q[18];
rx(1.5707963267948966) q[7];
fsim(1.5707963267948966, 0) q[22], q[3];
fsim(1.5707963267948966, 0) q[23], q[7];
rx(1.5707963267948966) q[17];
sw q[22];
sw q[4];
ry(1.5707963267948966) q[11];
sw q[26];
fsim(1.5707963267948966, 0) q[7], q[5];
sw q[11];
fsim(1.5707963267948966, 0) q[5], q[22];
rx(1.5707963267948966) q[14];
sw q[10];
fsim(1.5707963267948966, 0) q[26], q[18];
fsim(1.5707963267948966, 0) q[16], q[27];
ry(1.5707963267948966) q[22];
sw q[9];
rx(1.5707963267948966) q[13];
rx(1.5707963267948966) q[14];
ry(1.5707963267948966) q[27];
ry(1.5707963267948966) q[9];
sx q[3];
fsim(1.5707963267948966, 0) q[9], q[13];
sw q[0];
sx q[9];
sx q[27];
ry(1.5707963267948966) q[15];
ry(1.5707963267948966) q[21];
rx(1.5707963267948966) q[1];
fsim(1.5707963267948966, 0) q[17], q[0];
sw q[25];
sw q[22];
rx(1.5707963267948966) q[10];
sx q[14];
sw q[10];
sw q[12];
ry(1.5707963267948966) q[20];
sy q[17];
fsim(1.5707963267948966, 0) q[23], q[16];
fsim(1.5707963267948966, 0) q[19], q[25];
sy q[7];
rx(1.5707963267948966) q[6];
sw q[23];
ry(1.5707963267948966) q[1];
ry(1.5707963267948966) q[14];
rx(1.5707963267948966) q[15];
sy q[27];
sx q[22];
sw q[23];
sy q[20];
sw q[25];
ry(1.5707963267948966) q[16];
ry(1.5707963267948966) q[25];
rx(1.5707963267948966) q[18];
ry(1.5707963267948966) q[23];
ry(1.5707963267948966) q[21];
fsim(1.5707963267948966, 0) q[23], q[22];
ry(1.5707963267948966) q[13];
sx q[0];
fsim(1.5707963267948966, 0) q[11], q[23];
sx q[12];
rx(1.5707963267948966) q[11];
sy q[22];
fsim(1.5707963267948966, 0) q[15], q[1];
ry(1.5707963267948966) q[11];
sx q[7];
sx q[11];
ry(1.5707963267948966) q[20];
sw q[22];
ry(1.5707963267948966) q[23];
sw q[16];
fsim(1.5707963267948966, 0) q[11], q[0];
sx q[8];
sw q[10];
rx(1.5707963267948966) q[14];
sy q[6];
sw q[27];
sx q[20];
sy q[5];
rx(1.5707963267948966) q[2];
sw q[10];
fsim(1.5707963267948966, 0) q[26], q[18];
rx(1.5707963267948966) q[13];
sy q[5];
ry(1.5707963267948966) q[24];
sx q[5];
ry(1.5707963267948966) q[8];
ry(1.5707963267948966) q[17];
sw q[2];
sy q[12];
sy q[26];
fsim(1.5707963267948966, 0) q[24], q[3];
rx(1.5707963267948966) q[14];
sy q[15];
sy q[8];
fsim(1.5707963267948966, 0) q[4], q[17];
ry(1.5707963267948966) q[0];
sy q[25];
fsim(1.5707963267948966, 0) q[22], q[4];
rx(1.5707963267948966) q[19];
sw q[26];
ry(1.5707963267948966) q[17];
fsim(1.5707963267948966, 0) q[23], q[19];
sy q[20];
sx q[18];
sy q[21];
rx(1.5707963267948966) q[24];
fsim(1.5707963267948966, 0) q[16], q[22];
sw q[14];
sy q[26];
fsim(1.5707963267948966, 0) q[26], q[8];
sx q[16];
sx q[20];
rx(1.5707963267948966) q[6];
fsim(1.5707963267948966, 0) q[7], q[2];
ry(1.5707963267948966) q[8];
fsim(1.5707963267948966, 0) q[0], q[8];
fsim(1.5707963267948966, 0) q[5], q[25];
sw q[9];
sx q[10];
sw q[7];
sy q[21];
fsim(1.5707963267948966, 0) q[4], q[6];
fsim(1.5707963267948966, 0) q[25], q[12];
rx(1.5707963267948966) q[5];
rx(1.5707963267948966) q[17];

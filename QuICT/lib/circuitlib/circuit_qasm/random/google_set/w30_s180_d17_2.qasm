OPENQASM 2.0;
include "qelib1.inc";
qreg q[30];
creg c[30];
sy q[8];
sw q[23];
ry(1.5707963267948966) q[19];
ry(1.5707963267948966) q[22];
ry(1.5707963267948966) q[16];
sx q[7];
sy q[21];
fsim(1.5707963267948966, 0) q[24], q[23];
fsim(1.5707963267948966, 0) q[11], q[23];
fsim(1.5707963267948966, 0) q[9], q[15];
sx q[11];
sx q[7];
sy q[29];
sx q[18];
ry(1.5707963267948966) q[8];
rx(1.5707963267948966) q[8];
ry(1.5707963267948966) q[18];
sy q[21];
rx(1.5707963267948966) q[28];
sy q[17];
sx q[29];
ry(1.5707963267948966) q[17];
sy q[22];
fsim(1.5707963267948966, 0) q[3], q[19];
fsim(1.5707963267948966, 0) q[17], q[8];
rx(1.5707963267948966) q[24];
sy q[10];
sw q[28];
fsim(1.5707963267948966, 0) q[27], q[2];
ry(1.5707963267948966) q[13];
sw q[2];
rx(1.5707963267948966) q[23];
sw q[0];
rx(1.5707963267948966) q[14];
ry(1.5707963267948966) q[14];
sw q[10];
sx q[19];
sw q[7];
sx q[11];
fsim(1.5707963267948966, 0) q[7], q[0];
sw q[13];
ry(1.5707963267948966) q[16];
fsim(1.5707963267948966, 0) q[23], q[5];
fsim(1.5707963267948966, 0) q[10], q[22];
sw q[25];
fsim(1.5707963267948966, 0) q[29], q[12];
rx(1.5707963267948966) q[14];
fsim(1.5707963267948966, 0) q[22], q[24];
sx q[18];
sy q[23];
sy q[25];
ry(1.5707963267948966) q[9];
fsim(1.5707963267948966, 0) q[26], q[27];
fsim(1.5707963267948966, 0) q[2], q[8];
sw q[27];
sw q[23];
ry(1.5707963267948966) q[4];
rx(1.5707963267948966) q[16];
sx q[4];
sw q[8];
rx(1.5707963267948966) q[23];
ry(1.5707963267948966) q[29];
sw q[5];
rx(1.5707963267948966) q[29];
rx(1.5707963267948966) q[17];
rx(1.5707963267948966) q[19];
rx(1.5707963267948966) q[19];
sw q[14];
sx q[14];
sx q[29];
sx q[12];
rx(1.5707963267948966) q[23];
sy q[11];
sy q[19];
sy q[26];
ry(1.5707963267948966) q[6];
sx q[9];
sw q[23];
ry(1.5707963267948966) q[16];
ry(1.5707963267948966) q[20];
rx(1.5707963267948966) q[23];
sx q[25];
sx q[10];
sy q[23];
sw q[8];
sx q[10];
rx(1.5707963267948966) q[22];
rx(1.5707963267948966) q[11];
sy q[13];
sx q[18];
sw q[17];
ry(1.5707963267948966) q[5];
rx(1.5707963267948966) q[28];
sx q[3];
sw q[9];
ry(1.5707963267948966) q[21];
sx q[9];
ry(1.5707963267948966) q[26];
fsim(1.5707963267948966, 0) q[20], q[24];
sw q[12];
sw q[0];
rx(1.5707963267948966) q[9];
ry(1.5707963267948966) q[2];
sw q[12];
rx(1.5707963267948966) q[21];
fsim(1.5707963267948966, 0) q[16], q[24];
fsim(1.5707963267948966, 0) q[9], q[4];
sx q[2];
sy q[21];
sx q[2];
sx q[6];
sy q[5];
sy q[6];
fsim(1.5707963267948966, 0) q[16], q[26];
sy q[6];
sy q[10];
sy q[5];
sw q[27];
sx q[21];
fsim(1.5707963267948966, 0) q[14], q[22];
rx(1.5707963267948966) q[20];
sw q[11];
sy q[28];
rx(1.5707963267948966) q[4];
fsim(1.5707963267948966, 0) q[3], q[21];
ry(1.5707963267948966) q[23];
ry(1.5707963267948966) q[8];
ry(1.5707963267948966) q[5];
ry(1.5707963267948966) q[3];
sw q[4];
rx(1.5707963267948966) q[16];
sy q[13];
sy q[23];
rx(1.5707963267948966) q[5];
sy q[7];
sy q[18];
sx q[24];
fsim(1.5707963267948966, 0) q[6], q[28];
ry(1.5707963267948966) q[22];
ry(1.5707963267948966) q[21];
sx q[18];
sw q[9];
ry(1.5707963267948966) q[26];
sw q[12];
ry(1.5707963267948966) q[27];
sy q[28];
rx(1.5707963267948966) q[12];
rx(1.5707963267948966) q[19];
rx(1.5707963267948966) q[13];
sx q[13];
ry(1.5707963267948966) q[25];
fsim(1.5707963267948966, 0) q[6], q[16];
fsim(1.5707963267948966, 0) q[13], q[17];
fsim(1.5707963267948966, 0) q[6], q[5];
fsim(1.5707963267948966, 0) q[2], q[4];
ry(1.5707963267948966) q[15];
fsim(1.5707963267948966, 0) q[9], q[28];
rx(1.5707963267948966) q[15];
sw q[22];
fsim(1.5707963267948966, 0) q[29], q[24];
sw q[3];
sw q[7];
rx(1.5707963267948966) q[22];
sx q[4];
ry(1.5707963267948966) q[0];
sx q[4];
rx(1.5707963267948966) q[12];
sy q[28];
sw q[8];
sx q[14];
fsim(1.5707963267948966, 0) q[23], q[10];
sy q[26];
ry(1.5707963267948966) q[8];
sy q[29];
fsim(1.5707963267948966, 0) q[10], q[8];
sw q[23];
sy q[0];
sw q[8];
fsim(1.5707963267948966, 0) q[18], q[25];
ry(1.5707963267948966) q[23];

OPENQASM 2.0;
include "qelib1.inc";
qreg q[30];
creg c[30];
rx(1.5707963267948966) q[20];
ry(1.5707963267948966) q[16];
sy q[23];
ry(1.5707963267948966) q[20];
ry(1.5707963267948966) q[9];
sw q[19];
rx(1.5707963267948966) q[10];
sy q[27];
ry(1.5707963267948966) q[19];
sx q[9];
sx q[11];
fsim(1.5707963267948966, 0) q[28], q[21];
sw q[4];
sw q[16];
ry(1.5707963267948966) q[23];
fsim(1.5707963267948966, 0) q[15], q[21];
sw q[7];
fsim(1.5707963267948966, 0) q[19], q[6];
sy q[29];
ry(1.5707963267948966) q[23];
rx(1.5707963267948966) q[15];
sx q[20];
sy q[0];
sx q[1];
sx q[9];
ry(1.5707963267948966) q[25];
fsim(1.5707963267948966, 0) q[15], q[17];
sw q[26];
sw q[11];
fsim(1.5707963267948966, 0) q[11], q[7];
fsim(1.5707963267948966, 0) q[18], q[4];
sx q[28];
sw q[18];
fsim(1.5707963267948966, 0) q[11], q[20];
sx q[8];
rx(1.5707963267948966) q[0];
sy q[12];
rx(1.5707963267948966) q[5];
sx q[5];
ry(1.5707963267948966) q[21];
rx(1.5707963267948966) q[1];
rx(1.5707963267948966) q[29];
sw q[9];
sw q[15];
fsim(1.5707963267948966, 0) q[27], q[29];
ry(1.5707963267948966) q[11];
sw q[6];
sw q[28];
sy q[13];
ry(1.5707963267948966) q[27];
sx q[18];
sx q[27];
sx q[24];
fsim(1.5707963267948966, 0) q[10], q[7];
rx(1.5707963267948966) q[24];
rx(1.5707963267948966) q[20];
sx q[10];
sx q[8];
sy q[8];
rx(1.5707963267948966) q[2];
ry(1.5707963267948966) q[28];
sy q[14];
sy q[11];
sx q[14];
rx(1.5707963267948966) q[13];
sy q[14];
sx q[26];
ry(1.5707963267948966) q[24];
sw q[14];
ry(1.5707963267948966) q[15];
sx q[14];
fsim(1.5707963267948966, 0) q[1], q[11];
sy q[19];
sx q[14];
ry(1.5707963267948966) q[1];
ry(1.5707963267948966) q[14];
sw q[3];
fsim(1.5707963267948966, 0) q[12], q[21];
fsim(1.5707963267948966, 0) q[24], q[8];
rx(1.5707963267948966) q[29];
fsim(1.5707963267948966, 0) q[22], q[9];
fsim(1.5707963267948966, 0) q[13], q[11];
fsim(1.5707963267948966, 0) q[26], q[18];
sx q[16];
rx(1.5707963267948966) q[1];
ry(1.5707963267948966) q[25];
fsim(1.5707963267948966, 0) q[13], q[17];
sw q[2];
fsim(1.5707963267948966, 0) q[29], q[17];
sx q[27];
sy q[9];
ry(1.5707963267948966) q[21];
rx(1.5707963267948966) q[21];
sw q[11];
sx q[3];
ry(1.5707963267948966) q[15];
sx q[12];
sx q[24];
ry(1.5707963267948966) q[26];
sw q[28];
sw q[14];
fsim(1.5707963267948966, 0) q[8], q[17];
fsim(1.5707963267948966, 0) q[23], q[1];
sy q[29];
rx(1.5707963267948966) q[14];
ry(1.5707963267948966) q[13];
fsim(1.5707963267948966, 0) q[1], q[12];
ry(1.5707963267948966) q[12];
sy q[13];
sy q[12];
fsim(1.5707963267948966, 0) q[27], q[4];
sw q[23];
ry(1.5707963267948966) q[26];
sx q[2];
fsim(1.5707963267948966, 0) q[17], q[26];
sy q[7];
ry(1.5707963267948966) q[11];
fsim(1.5707963267948966, 0) q[1], q[28];
ry(1.5707963267948966) q[22];
sy q[13];
fsim(1.5707963267948966, 0) q[22], q[23];
sw q[20];
sw q[10];
fsim(1.5707963267948966, 0) q[20], q[18];
sy q[16];
sw q[2];
sx q[9];
sy q[8];
sy q[12];
sw q[24];
sw q[8];
fsim(1.5707963267948966, 0) q[11], q[28];
fsim(1.5707963267948966, 0) q[24], q[14];
fsim(1.5707963267948966, 0) q[11], q[20];
ry(1.5707963267948966) q[16];
sw q[5];
rx(1.5707963267948966) q[14];
sw q[7];
rx(1.5707963267948966) q[28];
fsim(1.5707963267948966, 0) q[25], q[29];
fsim(1.5707963267948966, 0) q[14], q[19];
sx q[8];
fsim(1.5707963267948966, 0) q[23], q[22];
fsim(1.5707963267948966, 0) q[15], q[18];
ry(1.5707963267948966) q[20];
sw q[8];
sw q[26];
sy q[0];
fsim(1.5707963267948966, 0) q[23], q[1];
ry(1.5707963267948966) q[7];
sy q[15];
sy q[2];
fsim(1.5707963267948966, 0) q[20], q[11];
sw q[18];
sy q[17];
rx(1.5707963267948966) q[6];
fsim(1.5707963267948966, 0) q[28], q[7];
fsim(1.5707963267948966, 0) q[8], q[10];
sx q[25];
ry(1.5707963267948966) q[8];
fsim(1.5707963267948966, 0) q[15], q[26];
ry(1.5707963267948966) q[29];
sw q[16];
sx q[17];
sy q[3];
rx(1.5707963267948966) q[28];
rx(1.5707963267948966) q[20];
fsim(1.5707963267948966, 0) q[27], q[12];
sy q[0];
fsim(1.5707963267948966, 0) q[9], q[26];
ry(1.5707963267948966) q[4];
fsim(1.5707963267948966, 0) q[26], q[6];
sw q[22];
rx(1.5707963267948966) q[10];
fsim(1.5707963267948966, 0) q[17], q[11];
fsim(1.5707963267948966, 0) q[20], q[29];
fsim(1.5707963267948966, 0) q[5], q[2];
sy q[13];
sw q[28];
sx q[27];
rx(1.5707963267948966) q[6];
sy q[21];
sw q[3];
sy q[16];
sw q[1];
ry(1.5707963267948966) q[17];
rx(1.5707963267948966) q[26];
ry(1.5707963267948966) q[25];
rx(1.5707963267948966) q[13];
ry(1.5707963267948966) q[20];
sy q[20];
ry(1.5707963267948966) q[6];
sw q[1];
ry(1.5707963267948966) q[16];
fsim(1.5707963267948966, 0) q[22], q[25];
rx(1.5707963267948966) q[23];
sy q[17];
sw q[25];
fsim(1.5707963267948966, 0) q[12], q[19];
sx q[14];
sy q[18];
fsim(1.5707963267948966, 0) q[11], q[15];
sw q[0];
sy q[10];
sx q[24];
rx(1.5707963267948966) q[8];
ry(1.5707963267948966) q[12];
sy q[13];
fsim(1.5707963267948966, 0) q[2], q[10];
sy q[0];

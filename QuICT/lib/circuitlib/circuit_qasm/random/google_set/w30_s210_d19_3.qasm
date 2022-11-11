OPENQASM 2.0;
include "qelib1.inc";
qreg q[30];
creg c[30];
sy q[27];
sw q[5];
sw q[21];
ry(1.5707963267948966) q[4];
ry(1.5707963267948966) q[22];
fsim(1.5707963267948966, 0) q[18], q[8];
fsim(1.5707963267948966, 0) q[14], q[24];
sy q[7];
sy q[17];
ry(1.5707963267948966) q[28];
fsim(1.5707963267948966, 0) q[15], q[11];
ry(1.5707963267948966) q[6];
fsim(1.5707963267948966, 0) q[11], q[19];
rx(1.5707963267948966) q[7];
fsim(1.5707963267948966, 0) q[10], q[6];
sw q[29];
fsim(1.5707963267948966, 0) q[22], q[23];
sy q[22];
fsim(1.5707963267948966, 0) q[24], q[7];
rx(1.5707963267948966) q[6];
sy q[21];
fsim(1.5707963267948966, 0) q[4], q[6];
sx q[26];
sy q[0];
sw q[22];
sx q[28];
sx q[24];
ry(1.5707963267948966) q[18];
fsim(1.5707963267948966, 0) q[24], q[19];
ry(1.5707963267948966) q[4];
fsim(1.5707963267948966, 0) q[17], q[22];
sw q[13];
sw q[13];
sx q[9];
sy q[3];
fsim(1.5707963267948966, 0) q[14], q[20];
sw q[3];
sy q[1];
sw q[11];
sy q[20];
fsim(1.5707963267948966, 0) q[14], q[17];
sy q[25];
sy q[18];
fsim(1.5707963267948966, 0) q[22], q[5];
ry(1.5707963267948966) q[28];
sx q[17];
sy q[7];
fsim(1.5707963267948966, 0) q[6], q[24];
fsim(1.5707963267948966, 0) q[29], q[17];
sy q[11];
sy q[6];
sw q[21];
sx q[13];
sy q[11];
ry(1.5707963267948966) q[13];
sw q[13];
sx q[22];
rx(1.5707963267948966) q[28];
sx q[1];
sy q[16];
ry(1.5707963267948966) q[9];
fsim(1.5707963267948966, 0) q[0], q[24];
fsim(1.5707963267948966, 0) q[9], q[0];
ry(1.5707963267948966) q[9];
ry(1.5707963267948966) q[26];
rx(1.5707963267948966) q[24];
ry(1.5707963267948966) q[26];
sx q[5];
sx q[13];
rx(1.5707963267948966) q[1];
fsim(1.5707963267948966, 0) q[12], q[8];
ry(1.5707963267948966) q[14];
fsim(1.5707963267948966, 0) q[2], q[24];
fsim(1.5707963267948966, 0) q[8], q[26];
sw q[11];
fsim(1.5707963267948966, 0) q[19], q[23];
rx(1.5707963267948966) q[13];
sy q[21];
ry(1.5707963267948966) q[7];
fsim(1.5707963267948966, 0) q[16], q[15];
fsim(1.5707963267948966, 0) q[20], q[29];
sx q[22];
fsim(1.5707963267948966, 0) q[4], q[7];
sw q[14];
sy q[23];
sy q[5];
sw q[1];
sw q[24];
sx q[27];
sw q[0];
ry(1.5707963267948966) q[20];
fsim(1.5707963267948966, 0) q[10], q[13];
sx q[8];
rx(1.5707963267948966) q[28];
fsim(1.5707963267948966, 0) q[22], q[1];
ry(1.5707963267948966) q[14];
sw q[6];
ry(1.5707963267948966) q[20];
sw q[6];
ry(1.5707963267948966) q[13];
sw q[11];
rx(1.5707963267948966) q[9];
fsim(1.5707963267948966, 0) q[11], q[5];
sw q[15];
fsim(1.5707963267948966, 0) q[16], q[15];
sw q[11];
ry(1.5707963267948966) q[23];
sx q[2];
sw q[6];
rx(1.5707963267948966) q[28];
fsim(1.5707963267948966, 0) q[13], q[4];
rx(1.5707963267948966) q[11];
rx(1.5707963267948966) q[9];
fsim(1.5707963267948966, 0) q[7], q[23];
sw q[19];
ry(1.5707963267948966) q[14];
ry(1.5707963267948966) q[9];
sw q[18];
sw q[22];
sy q[18];
sy q[2];
sx q[10];
rx(1.5707963267948966) q[19];
ry(1.5707963267948966) q[24];
sw q[9];
ry(1.5707963267948966) q[2];
sx q[13];
sw q[24];
fsim(1.5707963267948966, 0) q[29], q[23];
ry(1.5707963267948966) q[24];
ry(1.5707963267948966) q[4];
ry(1.5707963267948966) q[20];
ry(1.5707963267948966) q[9];
sw q[23];
sw q[3];
ry(1.5707963267948966) q[10];
sy q[10];
ry(1.5707963267948966) q[5];
ry(1.5707963267948966) q[2];
rx(1.5707963267948966) q[19];
ry(1.5707963267948966) q[4];
sw q[6];
fsim(1.5707963267948966, 0) q[17], q[28];
rx(1.5707963267948966) q[14];
rx(1.5707963267948966) q[18];
sy q[17];
fsim(1.5707963267948966, 0) q[20], q[4];
sw q[5];
fsim(1.5707963267948966, 0) q[20], q[21];
rx(1.5707963267948966) q[2];
fsim(1.5707963267948966, 0) q[6], q[1];
sw q[26];
ry(1.5707963267948966) q[20];
sy q[17];
ry(1.5707963267948966) q[6];
fsim(1.5707963267948966, 0) q[13], q[19];
sy q[29];
sw q[12];
sx q[15];
ry(1.5707963267948966) q[28];
sw q[11];
rx(1.5707963267948966) q[1];
rx(1.5707963267948966) q[9];
sw q[29];
rx(1.5707963267948966) q[27];
sy q[15];
rx(1.5707963267948966) q[21];
fsim(1.5707963267948966, 0) q[3], q[22];
sw q[15];
sy q[27];
rx(1.5707963267948966) q[6];
sw q[6];
fsim(1.5707963267948966, 0) q[9], q[18];
sx q[14];
fsim(1.5707963267948966, 0) q[0], q[27];
fsim(1.5707963267948966, 0) q[18], q[3];
rx(1.5707963267948966) q[25];
sw q[9];
sx q[27];
fsim(1.5707963267948966, 0) q[1], q[11];
ry(1.5707963267948966) q[0];
sy q[29];
sy q[13];
ry(1.5707963267948966) q[1];
sx q[28];
rx(1.5707963267948966) q[3];
sw q[3];
ry(1.5707963267948966) q[4];
sw q[15];
rx(1.5707963267948966) q[24];
ry(1.5707963267948966) q[5];
sx q[24];
ry(1.5707963267948966) q[20];
ry(1.5707963267948966) q[26];
ry(1.5707963267948966) q[7];
fsim(1.5707963267948966, 0) q[28], q[19];
sw q[0];
ry(1.5707963267948966) q[15];
sy q[28];
sx q[26];
sx q[10];
ry(1.5707963267948966) q[21];
sx q[7];
sx q[25];
sy q[14];
ry(1.5707963267948966) q[8];
ry(1.5707963267948966) q[19];
sy q[12];
ry(1.5707963267948966) q[27];
sx q[16];

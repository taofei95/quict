OPENQASM 2.0;
include "qelib1.inc";
qreg q[23];
creg c[23];
sx q[15];
sw q[20];
sx q[1];
ry(1.5707963267948966) q[22];
fsim(1.5707963267948966, 0) q[14], q[10];
sy q[10];
fsim(1.5707963267948966, 0) q[20], q[9];
sx q[9];
fsim(1.5707963267948966, 0) q[8], q[11];
rx(1.5707963267948966) q[13];
rx(1.5707963267948966) q[12];
sy q[3];
ry(1.5707963267948966) q[2];
sy q[10];
ry(1.5707963267948966) q[6];
ry(1.5707963267948966) q[13];
ry(1.5707963267948966) q[12];
sx q[1];
fsim(1.5707963267948966, 0) q[16], q[20];
sx q[22];
fsim(1.5707963267948966, 0) q[16], q[3];
rx(1.5707963267948966) q[4];
sy q[8];
sy q[9];
fsim(1.5707963267948966, 0) q[8], q[15];
sx q[16];
fsim(1.5707963267948966, 0) q[20], q[6];
sy q[1];
sy q[21];
sy q[0];
fsim(1.5707963267948966, 0) q[13], q[16];
sy q[3];
sy q[1];
sx q[4];
sw q[22];
sy q[18];
rx(1.5707963267948966) q[1];
sw q[11];
sw q[19];
fsim(1.5707963267948966, 0) q[15], q[16];
sw q[3];
fsim(1.5707963267948966, 0) q[8], q[10];
rx(1.5707963267948966) q[10];
rx(1.5707963267948966) q[4];
sy q[21];
sy q[14];
fsim(1.5707963267948966, 0) q[15], q[7];
fsim(1.5707963267948966, 0) q[6], q[2];
sx q[20];
sy q[18];
fsim(1.5707963267948966, 0) q[14], q[20];
ry(1.5707963267948966) q[20];
sx q[5];
fsim(1.5707963267948966, 0) q[1], q[13];
rx(1.5707963267948966) q[12];
sx q[10];
ry(1.5707963267948966) q[18];
rx(1.5707963267948966) q[7];
ry(1.5707963267948966) q[3];
sx q[0];
rx(1.5707963267948966) q[1];
sx q[11];
sw q[15];
sw q[21];
fsim(1.5707963267948966, 0) q[2], q[7];
sy q[8];
sw q[9];
ry(1.5707963267948966) q[12];
rx(1.5707963267948966) q[3];
fsim(1.5707963267948966, 0) q[3], q[8];
sw q[14];
sx q[15];
fsim(1.5707963267948966, 0) q[18], q[6];
fsim(1.5707963267948966, 0) q[18], q[17];
sw q[11];
rx(1.5707963267948966) q[16];
fsim(1.5707963267948966, 0) q[8], q[7];
fsim(1.5707963267948966, 0) q[1], q[21];
sx q[12];
sx q[22];
sw q[12];
rx(1.5707963267948966) q[14];
ry(1.5707963267948966) q[4];
fsim(1.5707963267948966, 0) q[2], q[11];
sy q[7];
fsim(1.5707963267948966, 0) q[0], q[14];
sw q[19];
sw q[15];
fsim(1.5707963267948966, 0) q[9], q[22];
sy q[5];
rx(1.5707963267948966) q[4];
rx(1.5707963267948966) q[0];
sx q[22];
fsim(1.5707963267948966, 0) q[4], q[20];
sx q[15];
sx q[11];
sw q[16];
ry(1.5707963267948966) q[6];
fsim(1.5707963267948966, 0) q[21], q[19];
sw q[14];
sy q[10];
ry(1.5707963267948966) q[13];
sy q[11];
fsim(1.5707963267948966, 0) q[15], q[18];
sw q[1];
ry(1.5707963267948966) q[16];
sy q[22];
sw q[12];
sx q[3];
sw q[0];
ry(1.5707963267948966) q[13];
fsim(1.5707963267948966, 0) q[12], q[6];
rx(1.5707963267948966) q[8];
sw q[3];
sx q[17];
sy q[10];
sx q[21];
sx q[13];
fsim(1.5707963267948966, 0) q[4], q[5];
sy q[20];
sx q[11];
sx q[7];
sy q[5];
fsim(1.5707963267948966, 0) q[11], q[19];
sy q[20];
sx q[20];
sx q[8];
sx q[16];
fsim(1.5707963267948966, 0) q[18], q[1];
ry(1.5707963267948966) q[15];
sw q[14];
fsim(1.5707963267948966, 0) q[21], q[19];
fsim(1.5707963267948966, 0) q[4], q[6];
sy q[6];
sx q[10];
sy q[5];
rx(1.5707963267948966) q[7];
ry(1.5707963267948966) q[19];
sw q[15];
sx q[17];
sw q[0];
sx q[20];
sw q[15];
fsim(1.5707963267948966, 0) q[7], q[13];
rx(1.5707963267948966) q[11];
sw q[18];
rx(1.5707963267948966) q[9];
sy q[2];
fsim(1.5707963267948966, 0) q[9], q[6];
sy q[0];
fsim(1.5707963267948966, 0) q[11], q[16];
fsim(1.5707963267948966, 0) q[10], q[17];
ry(1.5707963267948966) q[4];
sx q[0];
fsim(1.5707963267948966, 0) q[13], q[15];
sw q[6];
ry(1.5707963267948966) q[12];
sw q[18];
sy q[14];
fsim(1.5707963267948966, 0) q[22], q[14];
sy q[1];
rx(1.5707963267948966) q[2];
rx(1.5707963267948966) q[4];
sw q[4];
ry(1.5707963267948966) q[7];
rx(1.5707963267948966) q[6];
fsim(1.5707963267948966, 0) q[2], q[3];
sy q[3];
sy q[16];
rx(1.5707963267948966) q[3];
rx(1.5707963267948966) q[16];
fsim(1.5707963267948966, 0) q[17], q[1];
rx(1.5707963267948966) q[4];
ry(1.5707963267948966) q[5];
ry(1.5707963267948966) q[21];
fsim(1.5707963267948966, 0) q[8], q[4];
fsim(1.5707963267948966, 0) q[18], q[3];
sw q[6];
rx(1.5707963267948966) q[18];
ry(1.5707963267948966) q[11];
sw q[6];
sx q[12];
rx(1.5707963267948966) q[12];
rx(1.5707963267948966) q[21];

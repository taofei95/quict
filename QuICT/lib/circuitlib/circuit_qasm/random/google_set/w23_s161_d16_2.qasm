OPENQASM 2.0;
include "qelib1.inc";
qreg q[23];
creg c[23];
rx(1.5707963267948966) q[1];
sy q[19];
fsim(1.5707963267948966, 0) q[17], q[6];
sw q[19];
sw q[12];
rx(1.5707963267948966) q[20];
fsim(1.5707963267948966, 0) q[2], q[3];
fsim(1.5707963267948966, 0) q[13], q[16];
sw q[1];
fsim(1.5707963267948966, 0) q[13], q[18];
sx q[2];
ry(1.5707963267948966) q[21];
rx(1.5707963267948966) q[14];
sw q[22];
sy q[20];
rx(1.5707963267948966) q[22];
sy q[18];
sy q[16];
rx(1.5707963267948966) q[4];
sx q[16];
rx(1.5707963267948966) q[7];
ry(1.5707963267948966) q[21];
ry(1.5707963267948966) q[3];
sw q[17];
sy q[1];
sx q[6];
sy q[19];
ry(1.5707963267948966) q[4];
ry(1.5707963267948966) q[1];
sy q[11];
fsim(1.5707963267948966, 0) q[6], q[12];
sy q[12];
rx(1.5707963267948966) q[11];
sy q[4];
sx q[10];
ry(1.5707963267948966) q[13];
fsim(1.5707963267948966, 0) q[5], q[17];
ry(1.5707963267948966) q[10];
rx(1.5707963267948966) q[2];
ry(1.5707963267948966) q[5];
rx(1.5707963267948966) q[20];
sy q[12];
sw q[3];
sx q[3];
sy q[10];
sy q[14];
rx(1.5707963267948966) q[8];
fsim(1.5707963267948966, 0) q[3], q[21];
rx(1.5707963267948966) q[21];
sy q[5];
sy q[16];
fsim(1.5707963267948966, 0) q[15], q[1];
sw q[5];
sw q[0];
rx(1.5707963267948966) q[20];
sw q[6];
sy q[11];
fsim(1.5707963267948966, 0) q[0], q[15];
sy q[0];
rx(1.5707963267948966) q[12];
rx(1.5707963267948966) q[0];
sx q[11];
sx q[2];
ry(1.5707963267948966) q[5];
sx q[17];
ry(1.5707963267948966) q[19];
sx q[8];
fsim(1.5707963267948966, 0) q[4], q[19];
sy q[10];
fsim(1.5707963267948966, 0) q[17], q[11];
fsim(1.5707963267948966, 0) q[3], q[18];
rx(1.5707963267948966) q[14];
sx q[9];
rx(1.5707963267948966) q[13];
sw q[6];
sy q[16];
fsim(1.5707963267948966, 0) q[7], q[18];
fsim(1.5707963267948966, 0) q[13], q[7];
sx q[6];
rx(1.5707963267948966) q[1];
rx(1.5707963267948966) q[6];
fsim(1.5707963267948966, 0) q[1], q[17];
sx q[9];
sx q[21];
sx q[7];
fsim(1.5707963267948966, 0) q[18], q[22];
sy q[10];
rx(1.5707963267948966) q[21];
ry(1.5707963267948966) q[22];
sx q[3];
ry(1.5707963267948966) q[0];
sw q[4];
sy q[5];
sx q[5];
sy q[19];
rx(1.5707963267948966) q[6];
sw q[14];
ry(1.5707963267948966) q[2];
sw q[11];
sy q[10];
sy q[21];
fsim(1.5707963267948966, 0) q[14], q[0];
sw q[2];
fsim(1.5707963267948966, 0) q[6], q[21];
rx(1.5707963267948966) q[1];
sx q[14];
ry(1.5707963267948966) q[11];
sx q[0];
fsim(1.5707963267948966, 0) q[11], q[4];
sw q[12];
fsim(1.5707963267948966, 0) q[16], q[12];
fsim(1.5707963267948966, 0) q[21], q[15];
rx(1.5707963267948966) q[22];
ry(1.5707963267948966) q[21];
ry(1.5707963267948966) q[10];
sx q[7];
sy q[11];
sy q[15];
fsim(1.5707963267948966, 0) q[18], q[7];
fsim(1.5707963267948966, 0) q[21], q[17];
rx(1.5707963267948966) q[13];
sy q[6];
sx q[16];
rx(1.5707963267948966) q[16];
rx(1.5707963267948966) q[11];
sw q[10];
sw q[12];
sw q[1];
rx(1.5707963267948966) q[21];
fsim(1.5707963267948966, 0) q[3], q[22];
fsim(1.5707963267948966, 0) q[5], q[14];
fsim(1.5707963267948966, 0) q[16], q[2];
ry(1.5707963267948966) q[20];
sy q[21];
ry(1.5707963267948966) q[1];
sw q[6];
rx(1.5707963267948966) q[4];
sx q[9];
ry(1.5707963267948966) q[11];
rx(1.5707963267948966) q[9];
fsim(1.5707963267948966, 0) q[17], q[10];
sy q[13];
sx q[10];
sw q[22];
rx(1.5707963267948966) q[16];
sy q[12];
ry(1.5707963267948966) q[7];
sw q[2];
sy q[0];
sx q[3];
fsim(1.5707963267948966, 0) q[11], q[8];
fsim(1.5707963267948966, 0) q[12], q[7];
sy q[10];
sx q[4];
ry(1.5707963267948966) q[19];
fsim(1.5707963267948966, 0) q[0], q[17];
rx(1.5707963267948966) q[6];
rx(1.5707963267948966) q[2];
sy q[2];
ry(1.5707963267948966) q[12];
rx(1.5707963267948966) q[4];

OPENQASM 2.0;
include "qelib1.inc";
qreg q[21];
creg c[21];
rx(1.5707963267948966) q[8];
rx(1.5707963267948966) q[4];
rx(1.5707963267948966) q[10];
fsim(1.5707963267948966, 0) q[2], q[17];
ry(1.5707963267948966) q[19];
fsim(1.5707963267948966, 0) q[12], q[3];
sx q[13];
sy q[2];
fsim(1.5707963267948966, 0) q[6], q[2];
rx(1.5707963267948966) q[12];
sw q[11];
sx q[0];
sx q[3];
ry(1.5707963267948966) q[8];
fsim(1.5707963267948966, 0) q[14], q[8];
fsim(1.5707963267948966, 0) q[16], q[6];
fsim(1.5707963267948966, 0) q[11], q[0];
sy q[5];
fsim(1.5707963267948966, 0) q[3], q[17];
sx q[14];
fsim(1.5707963267948966, 0) q[17], q[3];
rx(1.5707963267948966) q[20];
sw q[7];
sy q[19];
sy q[9];
sy q[15];
sx q[9];
sw q[2];
fsim(1.5707963267948966, 0) q[5], q[6];
ry(1.5707963267948966) q[10];
rx(1.5707963267948966) q[12];
fsim(1.5707963267948966, 0) q[4], q[19];
sy q[10];
sx q[5];
sy q[17];
sx q[12];
fsim(1.5707963267948966, 0) q[2], q[20];
fsim(1.5707963267948966, 0) q[11], q[20];
sy q[17];
rx(1.5707963267948966) q[12];
sy q[14];
sy q[10];
sx q[16];
rx(1.5707963267948966) q[5];
sw q[6];
fsim(1.5707963267948966, 0) q[4], q[10];
fsim(1.5707963267948966, 0) q[0], q[14];
rx(1.5707963267948966) q[2];
rx(1.5707963267948966) q[16];
rx(1.5707963267948966) q[8];
ry(1.5707963267948966) q[18];
sx q[5];
rx(1.5707963267948966) q[6];
fsim(1.5707963267948966, 0) q[12], q[19];
sw q[15];
rx(1.5707963267948966) q[17];
sx q[0];
sw q[19];
sw q[0];
rx(1.5707963267948966) q[11];
ry(1.5707963267948966) q[12];
ry(1.5707963267948966) q[7];
sx q[18];
rx(1.5707963267948966) q[1];
fsim(1.5707963267948966, 0) q[10], q[15];
ry(1.5707963267948966) q[14];
sy q[18];
ry(1.5707963267948966) q[2];
ry(1.5707963267948966) q[17];
sw q[11];
fsim(1.5707963267948966, 0) q[13], q[10];
fsim(1.5707963267948966, 0) q[14], q[11];
sy q[18];
fsim(1.5707963267948966, 0) q[9], q[7];
sy q[10];
fsim(1.5707963267948966, 0) q[4], q[20];
ry(1.5707963267948966) q[16];
rx(1.5707963267948966) q[8];
sw q[16];
ry(1.5707963267948966) q[16];
fsim(1.5707963267948966, 0) q[8], q[13];
sy q[12];
sy q[20];
sy q[4];
ry(1.5707963267948966) q[9];
fsim(1.5707963267948966, 0) q[19], q[0];
sx q[20];
fsim(1.5707963267948966, 0) q[2], q[20];
sx q[19];
ry(1.5707963267948966) q[5];
fsim(1.5707963267948966, 0) q[0], q[8];
sx q[8];
sy q[1];
sy q[18];
sy q[19];
ry(1.5707963267948966) q[15];
rx(1.5707963267948966) q[8];
rx(1.5707963267948966) q[10];
fsim(1.5707963267948966, 0) q[16], q[9];
ry(1.5707963267948966) q[9];
fsim(1.5707963267948966, 0) q[11], q[16];
sy q[2];
rx(1.5707963267948966) q[19];
rx(1.5707963267948966) q[20];
sw q[20];
sw q[1];
ry(1.5707963267948966) q[17];
ry(1.5707963267948966) q[9];
sx q[11];
sw q[6];
sw q[5];
sw q[20];
ry(1.5707963267948966) q[8];
ry(1.5707963267948966) q[16];
sy q[11];
sx q[5];
sw q[10];
sw q[0];
sx q[8];
rx(1.5707963267948966) q[3];
fsim(1.5707963267948966, 0) q[11], q[15];
sw q[11];
ry(1.5707963267948966) q[20];
sy q[6];
rx(1.5707963267948966) q[0];
ry(1.5707963267948966) q[4];
fsim(1.5707963267948966, 0) q[11], q[1];
sx q[5];
ry(1.5707963267948966) q[12];
sy q[5];
sx q[12];
fsim(1.5707963267948966, 0) q[11], q[18];
sy q[0];
fsim(1.5707963267948966, 0) q[20], q[9];
sy q[18];
fsim(1.5707963267948966, 0) q[5], q[18];
ry(1.5707963267948966) q[14];
rx(1.5707963267948966) q[13];
sw q[15];
fsim(1.5707963267948966, 0) q[6], q[11];
fsim(1.5707963267948966, 0) q[20], q[13];
sy q[10];
fsim(1.5707963267948966, 0) q[12], q[6];
ry(1.5707963267948966) q[9];
sx q[0];
sw q[5];
rx(1.5707963267948966) q[13];
fsim(1.5707963267948966, 0) q[19], q[1];
rx(1.5707963267948966) q[8];
sy q[20];
sy q[11];
ry(1.5707963267948966) q[12];
sx q[12];
fsim(1.5707963267948966, 0) q[12], q[0];
sy q[0];
rx(1.5707963267948966) q[12];
sw q[1];
rx(1.5707963267948966) q[14];
sx q[6];
ry(1.5707963267948966) q[11];
sy q[14];
sw q[12];
ry(1.5707963267948966) q[3];
ry(1.5707963267948966) q[8];
fsim(1.5707963267948966, 0) q[18], q[16];
ry(1.5707963267948966) q[19];
ry(1.5707963267948966) q[8];
ry(1.5707963267948966) q[19];

OPENQASM 2.0;
include "qelib1.inc";
qreg q[21];
creg c[21];
sy q[13];
ry(1.5707963267948966) q[20];
sx q[2];
fsim(1.5707963267948966, 0) q[11], q[2];
rx(1.5707963267948966) q[18];
sy q[8];
fsim(1.5707963267948966, 0) q[17], q[14];
fsim(1.5707963267948966, 0) q[4], q[10];
rx(1.5707963267948966) q[15];
sw q[8];
fsim(1.5707963267948966, 0) q[2], q[18];
sx q[20];
fsim(1.5707963267948966, 0) q[5], q[10];
sw q[3];
sw q[3];
ry(1.5707963267948966) q[6];
sx q[3];
fsim(1.5707963267948966, 0) q[14], q[9];
fsim(1.5707963267948966, 0) q[15], q[13];
fsim(1.5707963267948966, 0) q[5], q[12];
sx q[20];
sy q[10];
rx(1.5707963267948966) q[10];
rx(1.5707963267948966) q[11];
fsim(1.5707963267948966, 0) q[13], q[14];
sw q[14];
sw q[8];
sw q[13];
ry(1.5707963267948966) q[18];
sy q[20];
sx q[6];
sx q[11];
rx(1.5707963267948966) q[12];
fsim(1.5707963267948966, 0) q[16], q[10];
sy q[17];
sy q[16];
rx(1.5707963267948966) q[7];
ry(1.5707963267948966) q[7];
sx q[11];
sw q[15];
sy q[2];
sy q[16];
sy q[20];
sx q[13];
sw q[16];
sx q[14];
sx q[14];
rx(1.5707963267948966) q[1];
ry(1.5707963267948966) q[4];
fsim(1.5707963267948966, 0) q[7], q[19];
fsim(1.5707963267948966, 0) q[16], q[8];
sx q[14];
sw q[4];
sw q[3];
sw q[9];
sx q[12];
sy q[0];
fsim(1.5707963267948966, 0) q[12], q[13];
sy q[6];
ry(1.5707963267948966) q[6];
sw q[11];
fsim(1.5707963267948966, 0) q[5], q[10];
ry(1.5707963267948966) q[5];
rx(1.5707963267948966) q[11];
ry(1.5707963267948966) q[16];
sy q[14];
sx q[20];
ry(1.5707963267948966) q[18];
fsim(1.5707963267948966, 0) q[0], q[20];
rx(1.5707963267948966) q[19];
ry(1.5707963267948966) q[5];
fsim(1.5707963267948966, 0) q[14], q[0];
sy q[17];
sx q[11];
sy q[18];
rx(1.5707963267948966) q[15];
sw q[3];
sy q[20];
rx(1.5707963267948966) q[20];
fsim(1.5707963267948966, 0) q[3], q[6];
sy q[17];
rx(1.5707963267948966) q[10];
sx q[16];
ry(1.5707963267948966) q[8];
sx q[3];
sx q[2];
ry(1.5707963267948966) q[16];
rx(1.5707963267948966) q[13];
fsim(1.5707963267948966, 0) q[0], q[4];
sw q[3];
rx(1.5707963267948966) q[17];
sy q[14];
sx q[19];
sx q[7];
rx(1.5707963267948966) q[10];
fsim(1.5707963267948966, 0) q[10], q[5];
ry(1.5707963267948966) q[17];
ry(1.5707963267948966) q[2];
sy q[6];
rx(1.5707963267948966) q[0];
sy q[0];
ry(1.5707963267948966) q[12];
ry(1.5707963267948966) q[9];
sx q[2];
ry(1.5707963267948966) q[18];
fsim(1.5707963267948966, 0) q[20], q[15];
ry(1.5707963267948966) q[5];
sw q[10];
sy q[17];
fsim(1.5707963267948966, 0) q[1], q[8];
fsim(1.5707963267948966, 0) q[18], q[4];
fsim(1.5707963267948966, 0) q[15], q[1];
ry(1.5707963267948966) q[15];
sy q[7];
rx(1.5707963267948966) q[12];
sx q[16];
ry(1.5707963267948966) q[1];
ry(1.5707963267948966) q[5];
sx q[19];
ry(1.5707963267948966) q[9];
rx(1.5707963267948966) q[18];
rx(1.5707963267948966) q[14];
ry(1.5707963267948966) q[19];
sy q[16];
fsim(1.5707963267948966, 0) q[1], q[14];
ry(1.5707963267948966) q[17];

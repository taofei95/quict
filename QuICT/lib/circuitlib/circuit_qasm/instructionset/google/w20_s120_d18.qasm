OPENQASM 2.0;
include "qelib1.inc";
qreg q[20];
creg c[20];
rx(1.5707963267948966) q[8];
rx(1.5707963267948966) q[11];
ry(1.5707963267948966) q[9];
sy q[19];
rx(1.5707963267948966) q[15];
sx q[3];
fsim(1.5707963267948966, 0) q[2], q[5];
fsim(1.5707963267948966, 0) q[0], q[10];
rx(1.5707963267948966) q[4];
sw q[16];
sx q[6];
sx q[14];
sy q[19];
ry(1.5707963267948966) q[7];
sx q[8];
ry(1.5707963267948966) q[4];
rx(1.5707963267948966) q[11];
sy q[5];
ry(1.5707963267948966) q[3];
sy q[19];
rx(1.5707963267948966) q[3];
fsim(1.5707963267948966, 0) q[11], q[1];
rx(1.5707963267948966) q[3];
fsim(1.5707963267948966, 0) q[3], q[5];
sw q[4];
rx(1.5707963267948966) q[7];
sw q[9];
ry(1.5707963267948966) q[5];
rx(1.5707963267948966) q[3];
fsim(1.5707963267948966, 0) q[3], q[14];
fsim(1.5707963267948966, 0) q[16], q[11];
ry(1.5707963267948966) q[9];
fsim(1.5707963267948966, 0) q[9], q[4];
sx q[13];
rx(1.5707963267948966) q[13];
sy q[5];
ry(1.5707963267948966) q[10];
sw q[14];
sy q[1];
rx(1.5707963267948966) q[18];
ry(1.5707963267948966) q[19];
sx q[9];
sy q[12];
sy q[14];
ry(1.5707963267948966) q[16];
fsim(1.5707963267948966, 0) q[12], q[19];
sw q[12];
fsim(1.5707963267948966, 0) q[7], q[14];
sy q[13];
sw q[11];
sy q[14];
rx(1.5707963267948966) q[18];
sx q[11];
sy q[5];
rx(1.5707963267948966) q[11];
sy q[14];
sx q[8];
fsim(1.5707963267948966, 0) q[11], q[13];
sy q[17];
fsim(1.5707963267948966, 0) q[17], q[16];
sy q[6];
fsim(1.5707963267948966, 0) q[7], q[10];
fsim(1.5707963267948966, 0) q[11], q[6];
fsim(1.5707963267948966, 0) q[7], q[12];
sw q[1];
sw q[4];
fsim(1.5707963267948966, 0) q[3], q[1];
fsim(1.5707963267948966, 0) q[14], q[19];
fsim(1.5707963267948966, 0) q[10], q[0];
ry(1.5707963267948966) q[17];
ry(1.5707963267948966) q[0];
rx(1.5707963267948966) q[1];
sy q[19];
sy q[9];
sy q[13];
fsim(1.5707963267948966, 0) q[2], q[8];
fsim(1.5707963267948966, 0) q[12], q[9];
fsim(1.5707963267948966, 0) q[10], q[12];
rx(1.5707963267948966) q[0];
sw q[4];
rx(1.5707963267948966) q[15];
sx q[10];
sx q[9];
fsim(1.5707963267948966, 0) q[16], q[7];
sw q[18];
sx q[2];
fsim(1.5707963267948966, 0) q[1], q[8];
sx q[3];
ry(1.5707963267948966) q[0];
sx q[6];
sw q[7];
fsim(1.5707963267948966, 0) q[11], q[1];
sy q[11];
sy q[3];
rx(1.5707963267948966) q[16];
fsim(1.5707963267948966, 0) q[4], q[13];
sy q[9];
rx(1.5707963267948966) q[4];
sy q[13];
sw q[5];
sw q[3];
sw q[9];
sx q[17];
rx(1.5707963267948966) q[2];
sw q[6];
sw q[12];
sy q[8];
sw q[5];
rx(1.5707963267948966) q[8];
sx q[17];
sw q[12];
fsim(1.5707963267948966, 0) q[18], q[4];
ry(1.5707963267948966) q[9];
fsim(1.5707963267948966, 0) q[13], q[15];
fsim(1.5707963267948966, 0) q[10], q[8];
fsim(1.5707963267948966, 0) q[5], q[10];
rx(1.5707963267948966) q[1];
sw q[5];
ry(1.5707963267948966) q[16];
ry(1.5707963267948966) q[19];

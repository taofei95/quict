OPENQASM 2.0;
include "qelib1.inc";
qreg q[23];
creg c[23];
rx(1.5707963267948966) q[12];
fsim(1.5707963267948966, 0) q[4], q[18];
rx(1.5707963267948966) q[5];
ry(1.5707963267948966) q[19];
sw q[4];
fsim(1.5707963267948966, 0) q[0], q[1];
sx q[14];
rx(1.5707963267948966) q[3];
sy q[2];
sw q[13];
sx q[22];
ry(1.5707963267948966) q[11];
fsim(1.5707963267948966, 0) q[17], q[14];
sy q[2];
ry(1.5707963267948966) q[19];
ry(1.5707963267948966) q[3];
fsim(1.5707963267948966, 0) q[18], q[13];
fsim(1.5707963267948966, 0) q[5], q[18];
sx q[1];
rx(1.5707963267948966) q[15];
fsim(1.5707963267948966, 0) q[10], q[12];
sy q[16];
sw q[14];
sy q[1];
rx(1.5707963267948966) q[20];
sx q[16];
fsim(1.5707963267948966, 0) q[13], q[15];
ry(1.5707963267948966) q[1];
rx(1.5707963267948966) q[20];
sx q[0];
fsim(1.5707963267948966, 0) q[22], q[3];
fsim(1.5707963267948966, 0) q[10], q[16];
ry(1.5707963267948966) q[9];
sw q[8];
sx q[21];
fsim(1.5707963267948966, 0) q[4], q[12];
sx q[12];
fsim(1.5707963267948966, 0) q[11], q[4];
fsim(1.5707963267948966, 0) q[0], q[19];
sx q[17];
sw q[16];
ry(1.5707963267948966) q[16];
sw q[16];
sy q[10];
rx(1.5707963267948966) q[1];
rx(1.5707963267948966) q[5];
sx q[16];
sx q[20];
sx q[21];
rx(1.5707963267948966) q[18];
sw q[10];
sw q[21];
sx q[21];
fsim(1.5707963267948966, 0) q[0], q[16];
rx(1.5707963267948966) q[19];
sw q[19];
ry(1.5707963267948966) q[16];
ry(1.5707963267948966) q[15];
rx(1.5707963267948966) q[2];
sw q[10];
rx(1.5707963267948966) q[16];
fsim(1.5707963267948966, 0) q[13], q[2];
ry(1.5707963267948966) q[14];
sw q[19];
fsim(1.5707963267948966, 0) q[22], q[19];
fsim(1.5707963267948966, 0) q[15], q[10];
sw q[1];
sy q[14];
ry(1.5707963267948966) q[9];
sx q[14];
fsim(1.5707963267948966, 0) q[17], q[3];
sy q[16];
fsim(1.5707963267948966, 0) q[14], q[5];
fsim(1.5707963267948966, 0) q[7], q[16];
sw q[9];
fsim(1.5707963267948966, 0) q[7], q[22];
ry(1.5707963267948966) q[20];
ry(1.5707963267948966) q[17];
sx q[14];
sy q[19];
rx(1.5707963267948966) q[12];
ry(1.5707963267948966) q[14];
sx q[3];
ry(1.5707963267948966) q[13];
ry(1.5707963267948966) q[4];
sx q[15];
sw q[10];
rx(1.5707963267948966) q[10];
sy q[9];
sy q[15];
fsim(1.5707963267948966, 0) q[13], q[21];
rx(1.5707963267948966) q[10];
sw q[22];
fsim(1.5707963267948966, 0) q[14], q[19];
sx q[2];
sw q[19];
ry(1.5707963267948966) q[6];
rx(1.5707963267948966) q[16];
rx(1.5707963267948966) q[11];
fsim(1.5707963267948966, 0) q[2], q[18];
sx q[5];
rx(1.5707963267948966) q[17];
sw q[15];
sy q[14];
sy q[13];
sw q[18];
sx q[8];
sw q[5];
fsim(1.5707963267948966, 0) q[11], q[5];
ry(1.5707963267948966) q[17];
fsim(1.5707963267948966, 0) q[11], q[15];
sx q[17];
sx q[11];
fsim(1.5707963267948966, 0) q[1], q[10];
ry(1.5707963267948966) q[8];
ry(1.5707963267948966) q[22];
rx(1.5707963267948966) q[12];
sx q[18];
ry(1.5707963267948966) q[8];
ry(1.5707963267948966) q[10];
sx q[18];
fsim(1.5707963267948966, 0) q[17], q[20];
sw q[7];
sw q[12];
ry(1.5707963267948966) q[12];
fsim(1.5707963267948966, 0) q[17], q[11];
sw q[4];
sy q[18];
sx q[0];
rx(1.5707963267948966) q[9];
sw q[3];
rx(1.5707963267948966) q[21];
fsim(1.5707963267948966, 0) q[20], q[21];
sw q[10];
sw q[18];
rx(1.5707963267948966) q[17];
fsim(1.5707963267948966, 0) q[20], q[18];
rx(1.5707963267948966) q[16];

OPENQASM 2.0;
include "qelib1.inc";
qreg q[26];
creg c[26];
sx q[19];
rz(1.5707963267948966) q[6];
x q[21];
cx q[0], q[4];
sx q[8];
rz(1.5707963267948966) q[11];
cx q[4], q[19];
rz(1.5707963267948966) q[18];
sx q[3];
x q[6];
x q[25];
rz(1.5707963267948966) q[14];
cx q[13], q[22];
rz(1.5707963267948966) q[14];
cx q[24], q[2];
x q[15];
sx q[8];
cx q[22], q[13];
x q[18];
rz(1.5707963267948966) q[19];
rz(1.5707963267948966) q[9];
sx q[4];
cx q[7], q[18];
rz(1.5707963267948966) q[15];
rz(1.5707963267948966) q[19];
cx q[25], q[7];
sx q[13];
cx q[12], q[21];
x q[24];
sx q[11];
x q[3];
rz(1.5707963267948966) q[21];
x q[15];
rz(1.5707963267948966) q[23];
sx q[16];
sx q[11];
cx q[18], q[21];
cx q[19], q[15];
x q[19];
rz(1.5707963267948966) q[19];
x q[23];
cx q[18], q[9];
sx q[7];
sx q[10];
cx q[11], q[10];
sx q[11];
cx q[2], q[3];
rz(1.5707963267948966) q[17];
sx q[3];
x q[11];
rz(1.5707963267948966) q[8];
cx q[0], q[10];
rz(1.5707963267948966) q[20];
rz(1.5707963267948966) q[0];
sx q[5];
rz(1.5707963267948966) q[17];
x q[3];
rz(1.5707963267948966) q[25];
x q[22];
sx q[19];
rz(1.5707963267948966) q[15];
cx q[14], q[9];
x q[7];
sx q[17];
sx q[0];
x q[19];
x q[0];
x q[3];
rz(1.5707963267948966) q[24];
cx q[6], q[11];
x q[17];
cx q[11], q[24];
sx q[5];
sx q[22];
x q[21];
sx q[8];
sx q[7];
rz(1.5707963267948966) q[22];
sx q[3];
sx q[12];
rz(1.5707963267948966) q[16];
sx q[16];
sx q[5];
cx q[19], q[21];
rz(1.5707963267948966) q[22];
rz(1.5707963267948966) q[12];
rz(1.5707963267948966) q[6];
sx q[6];
sx q[9];
x q[17];
sx q[15];
sx q[22];
cx q[15], q[5];
rz(1.5707963267948966) q[16];
rz(1.5707963267948966) q[25];
rz(1.5707963267948966) q[2];
sx q[24];
sx q[23];
cx q[11], q[4];
cx q[23], q[12];
rz(1.5707963267948966) q[18];
cx q[22], q[17];
sx q[21];
rz(1.5707963267948966) q[10];
cx q[1], q[22];
sx q[22];
rz(1.5707963267948966) q[15];
cx q[3], q[23];
sx q[18];
rz(1.5707963267948966) q[25];
x q[10];
rz(1.5707963267948966) q[2];
rz(1.5707963267948966) q[0];
x q[18];
x q[25];
rz(1.5707963267948966) q[6];
sx q[19];
sx q[7];
sx q[13];
sx q[12];
cx q[22], q[16];
cx q[13], q[8];
cx q[22], q[4];
cx q[12], q[1];
rz(1.5707963267948966) q[1];
sx q[20];
x q[9];
rz(1.5707963267948966) q[4];
sx q[2];
sx q[1];
x q[24];
cx q[15], q[21];
sx q[1];
cx q[22], q[8];
sx q[9];
sx q[2];
x q[10];
x q[15];
rz(1.5707963267948966) q[11];
x q[7];
sx q[18];
x q[7];
cx q[18], q[1];
sx q[16];
rz(1.5707963267948966) q[2];
rz(1.5707963267948966) q[24];
cx q[0], q[7];
rz(1.5707963267948966) q[19];
x q[20];
x q[3];
sx q[11];
rz(1.5707963267948966) q[10];
cx q[21], q[22];
sx q[0];
x q[7];
rz(1.5707963267948966) q[2];
rz(1.5707963267948966) q[0];
x q[17];
rz(1.5707963267948966) q[11];
x q[0];
rz(1.5707963267948966) q[18];
sx q[6];
cx q[21], q[5];
sx q[15];
rz(1.5707963267948966) q[8];
sx q[1];
rz(1.5707963267948966) q[0];
x q[25];
rz(1.5707963267948966) q[14];
x q[2];
sx q[11];
x q[12];
sx q[18];
cx q[15], q[4];
cx q[12], q[0];
sx q[20];
x q[22];
x q[3];
rz(1.5707963267948966) q[0];
x q[10];
rz(1.5707963267948966) q[17];
rz(1.5707963267948966) q[19];

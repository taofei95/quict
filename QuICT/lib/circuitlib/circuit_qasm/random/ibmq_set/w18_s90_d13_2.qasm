OPENQASM 2.0;
include "qelib1.inc";
qreg q[18];
creg c[18];
x q[5];
x q[4];
cx q[7], q[3];
x q[17];
rz(1.5707963267948966) q[12];
rz(1.5707963267948966) q[4];
rz(1.5707963267948966) q[6];
x q[0];
cx q[8], q[14];
cx q[2], q[15];
cx q[3], q[12];
rz(1.5707963267948966) q[10];
cx q[16], q[5];
sx q[0];
sx q[11];
rz(1.5707963267948966) q[13];
x q[17];
rz(1.5707963267948966) q[11];
x q[13];
sx q[13];
sx q[17];
sx q[4];
x q[11];
rz(1.5707963267948966) q[8];
sx q[0];
x q[7];
sx q[4];
rz(1.5707963267948966) q[15];
sx q[7];
sx q[4];
sx q[6];
x q[0];
cx q[13], q[17];
rz(1.5707963267948966) q[12];
sx q[3];
sx q[2];
x q[10];
rz(1.5707963267948966) q[12];
rz(1.5707963267948966) q[6];
rz(1.5707963267948966) q[11];
cx q[4], q[15];
rz(1.5707963267948966) q[0];
x q[7];
x q[0];
rz(1.5707963267948966) q[3];
sx q[8];
rz(1.5707963267948966) q[10];
sx q[10];
cx q[11], q[3];
rz(1.5707963267948966) q[5];
rz(1.5707963267948966) q[4];
rz(1.5707963267948966) q[8];
x q[14];
x q[11];
sx q[4];
sx q[10];
sx q[15];
rz(1.5707963267948966) q[17];
x q[1];
cx q[7], q[17];
rz(1.5707963267948966) q[12];
cx q[3], q[8];
cx q[5], q[0];
cx q[5], q[1];
rz(1.5707963267948966) q[0];
sx q[10];
sx q[7];
sx q[0];
x q[16];
sx q[17];
sx q[6];
cx q[17], q[0];
sx q[1];
x q[0];
x q[1];
cx q[17], q[9];
sx q[6];
x q[6];
cx q[13], q[9];
x q[5];
sx q[15];
rz(1.5707963267948966) q[11];
x q[9];
sx q[6];
rz(1.5707963267948966) q[11];
sx q[17];
cx q[2], q[0];
rz(1.5707963267948966) q[10];
x q[12];
cx q[7], q[2];

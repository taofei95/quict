OPENQASM 2.0;
include "qelib1.inc";
qreg q[15];
creg c[15];
rz(1.5707963267948966) q[4];
rz(1.5707963267948966) q[6];
x q[5];
cx q[4], q[12];
sx q[8];
rz(1.5707963267948966) q[2];
rz(1.5707963267948966) q[7];
x q[3];
sx q[5];
cx q[7], q[6];
x q[5];
rz(1.5707963267948966) q[1];
sx q[4];
cx q[1], q[10];
x q[9];
sx q[13];
cx q[5], q[4];
sx q[3];
rz(1.5707963267948966) q[3];
x q[6];
x q[12];
sx q[3];
x q[10];
sx q[7];
cx q[8], q[7];
sx q[11];
cx q[14], q[2];
rz(1.5707963267948966) q[14];
cx q[6], q[5];
rz(1.5707963267948966) q[12];
sx q[1];
rz(1.5707963267948966) q[8];
rz(1.5707963267948966) q[9];
x q[3];
x q[6];
rz(1.5707963267948966) q[1];
cx q[9], q[1];
x q[3];
rz(1.5707963267948966) q[3];
cx q[11], q[1];
cx q[5], q[13];
rz(1.5707963267948966) q[4];
cx q[9], q[1];
rz(1.5707963267948966) q[13];
sx q[3];
sx q[7];
cx q[11], q[9];
cx q[7], q[8];
rz(1.5707963267948966) q[2];
rz(1.5707963267948966) q[7];
sx q[10];
cx q[5], q[3];
x q[7];
rz(1.5707963267948966) q[4];
x q[5];
rz(1.5707963267948966) q[14];
rz(1.5707963267948966) q[5];
cx q[7], q[4];
x q[9];
cx q[12], q[9];
x q[12];
x q[2];
sx q[7];
cx q[12], q[7];
rz(1.5707963267948966) q[0];
rz(1.5707963267948966) q[2];
cx q[6], q[3];
sx q[2];
x q[9];
x q[11];
x q[1];
cx q[4], q[9];
rz(1.5707963267948966) q[3];
sx q[2];
x q[11];
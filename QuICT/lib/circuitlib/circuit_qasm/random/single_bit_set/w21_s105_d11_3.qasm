OPENQASM 2.0;
include "qelib1.inc";
qreg q[21];
creg c[21];
t q[8];
x q[20];
s q[13];
h q[13];
u1(1.5707963267948966) q[13];
z q[6];
u2(1.5707963267948966, 1.5707963267948966) q[3];
tdg q[9];
u3(0, 0, 1.5707963267948966) q[7];
z q[11];
ry(1.5707963267948966) q[7];
x q[3];
h q[20];
h q[19];
z q[4];
rz(1.5707963267948966) q[3];
tdg q[9];
u2(1.5707963267948966, 1.5707963267948966) q[10];
u1(1.5707963267948966) q[14];
rx(1.5707963267948966) q[11];
u3(0, 0, 1.5707963267948966) q[10];
y q[5];
ry(1.5707963267948966) q[17];
x q[0];
z q[9];
tdg q[18];
tdg q[5];
ry(1.5707963267948966) q[2];
tdg q[4];
t q[16];
rz(1.5707963267948966) q[8];
y q[9];
y q[12];
z q[3];
rx(1.5707963267948966) q[13];
x q[19];
x q[5];
s q[16];
u2(1.5707963267948966, 1.5707963267948966) q[7];
x q[16];
t q[14];
x q[9];
u1(1.5707963267948966) q[19];
s q[13];
s q[9];
h q[12];
x q[9];
t q[6];
ry(1.5707963267948966) q[8];
rx(1.5707963267948966) q[1];
x q[11];
x q[3];
z q[15];
s q[16];
y q[9];
s q[0];
y q[0];
s q[20];
t q[18];
u3(0, 0, 1.5707963267948966) q[14];
h q[14];
x q[19];
u2(1.5707963267948966, 1.5707963267948966) q[12];
h q[1];
h q[0];
rz(1.5707963267948966) q[6];
z q[3];
h q[15];
y q[12];
h q[4];
u2(1.5707963267948966, 1.5707963267948966) q[17];
t q[11];
s q[4];
t q[13];
tdg q[7];
rz(1.5707963267948966) q[10];
rz(1.5707963267948966) q[19];
z q[0];
h q[2];
y q[3];
tdg q[11];
u1(1.5707963267948966) q[9];
sdg q[16];
u2(1.5707963267948966, 1.5707963267948966) q[4];
rz(1.5707963267948966) q[16];
u2(1.5707963267948966, 1.5707963267948966) q[6];
y q[9];
tdg q[14];
y q[9];
u2(1.5707963267948966, 1.5707963267948966) q[15];
s q[4];
rx(1.5707963267948966) q[15];
y q[17];
u1(1.5707963267948966) q[3];
u3(0, 0, 1.5707963267948966) q[10];
y q[7];
u1(1.5707963267948966) q[16];
u1(1.5707963267948966) q[18];
y q[10];
ry(1.5707963267948966) q[13];
u3(0, 0, 1.5707963267948966) q[19];
rz(1.5707963267948966) q[5];
x q[11];
x q[6];
tdg q[16];

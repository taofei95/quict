OPENQASM 2.0;
include "qelib1.inc";
qreg q[21];
creg c[21];
h q[20];
x q[19];
t q[12];
rx(1.5707963267948966) q[10];
ry(1.5707963267948966) q[10];
u3(0, 0, 1.5707963267948966) q[6];
s q[6];
rz(1.5707963267948966) q[7];
u1(1.5707963267948966) q[4];
rx(1.5707963267948966) q[0];
rx(1.5707963267948966) q[18];
s q[1];
rx(1.5707963267948966) q[17];
u1(1.5707963267948966) q[3];
tdg q[13];
ry(1.5707963267948966) q[8];
u1(1.5707963267948966) q[3];
s q[6];
x q[20];
u2(1.5707963267948966, 1.5707963267948966) q[0];
y q[6];
tdg q[20];
u2(1.5707963267948966, 1.5707963267948966) q[5];
u3(0, 0, 1.5707963267948966) q[14];
t q[1];
ry(1.5707963267948966) q[7];
h q[17];
rz(1.5707963267948966) q[18];
s q[5];
h q[18];
z q[17];
u3(0, 0, 1.5707963267948966) q[5];
u2(1.5707963267948966, 1.5707963267948966) q[6];
s q[12];
x q[5];
h q[0];
y q[0];
ry(1.5707963267948966) q[13];
sdg q[12];
u2(1.5707963267948966, 1.5707963267948966) q[17];
u3(0, 0, 1.5707963267948966) q[3];
x q[6];
y q[9];
ry(1.5707963267948966) q[1];
u3(0, 0, 1.5707963267948966) q[20];
ry(1.5707963267948966) q[20];
u2(1.5707963267948966, 1.5707963267948966) q[13];
u1(1.5707963267948966) q[9];
rx(1.5707963267948966) q[2];
rz(1.5707963267948966) q[6];
t q[20];
t q[3];
u1(1.5707963267948966) q[11];
tdg q[15];
x q[10];
ry(1.5707963267948966) q[6];
x q[0];
x q[1];
s q[15];
sdg q[13];
s q[14];
rz(1.5707963267948966) q[17];
y q[19];
tdg q[9];
sdg q[1];
z q[3];
tdg q[13];
u3(0, 0, 1.5707963267948966) q[19];
z q[19];
u2(1.5707963267948966, 1.5707963267948966) q[8];
u2(1.5707963267948966, 1.5707963267948966) q[4];
y q[16];
s q[2];
ry(1.5707963267948966) q[2];
sdg q[6];
y q[19];
rx(1.5707963267948966) q[20];
rx(1.5707963267948966) q[14];
t q[17];
u1(1.5707963267948966) q[10];
z q[10];
u1(1.5707963267948966) q[8];
u3(0, 0, 1.5707963267948966) q[4];
u3(0, 0, 1.5707963267948966) q[8];
y q[9];
s q[19];
rz(1.5707963267948966) q[11];
sdg q[12];
tdg q[5];
tdg q[5];
h q[16];
t q[20];
h q[2];
ry(1.5707963267948966) q[14];
tdg q[19];
t q[12];
x q[19];
x q[15];
x q[18];
h q[20];
tdg q[2];
u2(1.5707963267948966, 1.5707963267948966) q[0];
rx(1.5707963267948966) q[11];
z q[3];
t q[1];
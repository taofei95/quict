OPENQASM 2.0;
include "qelib1.inc";
qreg q[23];
creg c[23];
rx(1.5707963267948966) q[6];
x q[14];
x q[9];
ry(1.5707963267948966) q[5];
s q[15];
rz(1.5707963267948966) q[11];
tdg q[6];
z q[2];
u1(1.5707963267948966) q[21];
u3(0, 0, 1.5707963267948966) q[10];
y q[19];
y q[22];
y q[15];
y q[15];
sdg q[18];
tdg q[20];
y q[7];
u3(0, 0, 1.5707963267948966) q[3];
tdg q[9];
z q[13];
sdg q[5];
y q[9];
y q[19];
z q[10];
ry(1.5707963267948966) q[6];
ry(1.5707963267948966) q[18];
u1(1.5707963267948966) q[22];
t q[4];
s q[2];
z q[16];
t q[11];
u2(1.5707963267948966, 1.5707963267948966) q[8];
u2(1.5707963267948966, 1.5707963267948966) q[8];
ry(1.5707963267948966) q[16];
u2(1.5707963267948966, 1.5707963267948966) q[13];
t q[18];
x q[1];
u1(1.5707963267948966) q[16];
z q[20];
y q[7];
y q[15];
u2(1.5707963267948966, 1.5707963267948966) q[7];
h q[0];
z q[13];
t q[17];
rz(1.5707963267948966) q[5];
t q[11];
z q[6];
u2(1.5707963267948966, 1.5707963267948966) q[10];
ry(1.5707963267948966) q[4];
u2(1.5707963267948966, 1.5707963267948966) q[4];
x q[1];
rx(1.5707963267948966) q[20];
y q[20];
tdg q[18];
x q[6];
h q[16];
y q[15];
sdg q[2];
x q[16];
rz(1.5707963267948966) q[17];
tdg q[18];
s q[14];
h q[5];
t q[8];
rz(1.5707963267948966) q[10];
s q[13];
z q[11];
y q[18];
t q[1];
x q[10];
ry(1.5707963267948966) q[17];
s q[7];
tdg q[18];
sdg q[5];
u1(1.5707963267948966) q[5];
sdg q[10];
u3(0, 0, 1.5707963267948966) q[18];
ry(1.5707963267948966) q[4];
rx(1.5707963267948966) q[12];
u2(1.5707963267948966, 1.5707963267948966) q[21];
t q[21];
s q[16];
h q[12];
x q[8];
sdg q[7];
u3(0, 0, 1.5707963267948966) q[7];
tdg q[2];
z q[5];
z q[1];
tdg q[18];
ry(1.5707963267948966) q[13];
s q[10];
rz(1.5707963267948966) q[6];
tdg q[11];
sdg q[3];
rx(1.5707963267948966) q[17];
u1(1.5707963267948966) q[16];
h q[19];
tdg q[19];
tdg q[14];
t q[18];
x q[22];
u2(1.5707963267948966, 1.5707963267948966) q[19];
s q[3];
rz(1.5707963267948966) q[1];
ry(1.5707963267948966) q[5];
sdg q[12];
u3(0, 0, 1.5707963267948966) q[18];
sdg q[6];
h q[13];
rx(1.5707963267948966) q[4];
t q[11];
u1(1.5707963267948966) q[19];
tdg q[6];
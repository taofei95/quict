OPENQASM 2.0;
include "qelib1.inc";
qreg q[22];
creg c[22];
y q[17];
sdg q[14];
sdg q[14];
u1(1.5707963267948966) q[8];
h q[18];
tdg q[14];
tdg q[5];
tdg q[9];
z q[19];
sdg q[7];
ry(1.5707963267948966) q[21];
u3(0, 0, 1.5707963267948966) q[9];
x q[17];
t q[0];
s q[12];
ry(1.5707963267948966) q[20];
y q[19];
z q[16];
t q[12];
t q[20];
tdg q[5];
rx(1.5707963267948966) q[15];
x q[18];
tdg q[10];
u3(0, 0, 1.5707963267948966) q[11];
y q[0];
x q[1];
u1(1.5707963267948966) q[13];
z q[3];
u1(1.5707963267948966) q[21];
u1(1.5707963267948966) q[20];
ry(1.5707963267948966) q[1];
tdg q[17];
sdg q[12];
ry(1.5707963267948966) q[13];
y q[19];
z q[10];
z q[14];
z q[4];
u3(0, 0, 1.5707963267948966) q[21];
x q[16];
rx(1.5707963267948966) q[6];
z q[2];
tdg q[9];
s q[14];
z q[17];
z q[1];
rx(1.5707963267948966) q[2];
tdg q[20];
u2(1.5707963267948966, 1.5707963267948966) q[15];
tdg q[0];
h q[4];
u1(1.5707963267948966) q[12];
tdg q[15];
u2(1.5707963267948966, 1.5707963267948966) q[4];
t q[19];
t q[19];
h q[16];
rz(1.5707963267948966) q[16];
u1(1.5707963267948966) q[11];
h q[16];
s q[5];
u3(0, 0, 1.5707963267948966) q[1];
u1(1.5707963267948966) q[3];
s q[19];
sdg q[8];
y q[18];
t q[7];
rz(1.5707963267948966) q[10];
rx(1.5707963267948966) q[2];
u1(1.5707963267948966) q[12];
u3(0, 0, 1.5707963267948966) q[0];
ry(1.5707963267948966) q[10];
rz(1.5707963267948966) q[1];
h q[16];
u3(0, 0, 1.5707963267948966) q[16];
u2(1.5707963267948966, 1.5707963267948966) q[13];
rx(1.5707963267948966) q[19];
rz(1.5707963267948966) q[15];
x q[16];
t q[6];
u3(0, 0, 1.5707963267948966) q[16];
tdg q[20];
rz(1.5707963267948966) q[4];
t q[4];
tdg q[8];
z q[6];
y q[18];
x q[20];
rz(1.5707963267948966) q[9];
t q[6];
tdg q[0];
x q[0];
u1(1.5707963267948966) q[4];
u1(1.5707963267948966) q[16];
s q[1];
z q[6];
rz(1.5707963267948966) q[0];
tdg q[11];
h q[20];
rz(1.5707963267948966) q[11];
y q[3];
sdg q[11];
u2(1.5707963267948966, 1.5707963267948966) q[14];
z q[21];
x q[8];
x q[20];
u2(1.5707963267948966, 1.5707963267948966) q[11];
tdg q[6];
y q[19];
sdg q[16];
u2(1.5707963267948966, 1.5707963267948966) q[3];
rz(1.5707963267948966) q[5];
u2(1.5707963267948966, 1.5707963267948966) q[16];
ry(1.5707963267948966) q[17];
u1(1.5707963267948966) q[1];
z q[5];
rx(1.5707963267948966) q[4];
x q[9];
ry(1.5707963267948966) q[18];
h q[4];
tdg q[8];
s q[11];
ry(1.5707963267948966) q[2];
u2(1.5707963267948966, 1.5707963267948966) q[15];
u3(0, 0, 1.5707963267948966) q[4];
sdg q[16];
u1(1.5707963267948966) q[17];
tdg q[16];
u3(0, 0, 1.5707963267948966) q[7];
tdg q[9];
h q[18];
